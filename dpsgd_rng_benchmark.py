import torch
import torch.nn as nn
from torch.func import vmap, grad, functional_call
import time
from functools import wraps
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import psutil
import numpy as np
from contextlib import contextmanager
import statistics
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from model import DPTransformerEncoder, DPResNet, ModelConfig, create_dp_resnet18
from opacus.optimizers.optimizer import _generate_noise
import opacus.optimizers.optimizer as opacus_opt
from opacus.utils.batch_memory_manager import BatchMemoryManager

@dataclass
class BenchmarkConfig(ModelConfig):
    """Configuration for benchmark runs, inheriting from ModelConfig"""
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    num_iterations: int = 100
    warmup_iterations: int = 10
    poisson_sampling: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@contextmanager
def timer(name: str, stats_dict: Dict[str, list], device: str):
    """Enhanced timer context manager with proper GPU synchronization"""
    # Synchronize before starting timer
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    yield
    
    # Synchronize before stopping timer
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
        
    end = time.perf_counter()
    duration_ms = (end - start) * 1000
    if name not in stats_dict:
        stats_dict[name] = []
    stats_dict[name].append(duration_ms)

class DPSGDBenchmark:
    """Benchmark different variants of SGD training"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.timing_stats = {}

        def synchronize_device(device):
            """Helper function to synchronize based on device"""
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
                
        self.synchronize_device = synchronize_device  # Save as instance method

        # Monkey patch opacus noise generation to get timing
        # Create wrapper with its own timing stats
        def timed_noise_generation(func):
            """Enhanced wrapper for noise generation timing with std debugging"""
            noise_stats = {
                'noise_generation_zero': [],
                'noise_generation_nonzero': []
            }
            
            # Static variable to control printing
            first_call = True
            
            @wraps(func)
            def wrapper(std: float, *args, **kwargs):
                nonlocal first_call
                device = args[0].device.type if args else kwargs.get('device', 'cpu')
                
                # Print std value only on first call
                if first_call:
                    print(f"\nNoise Generation std parameter: {std}")
                    first_call = False
                
                # Synchronize before starting
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                    
                start = time.perf_counter()
                result = func(std, *args, **kwargs)
                
                # Synchronize after noise generation
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                    
                duration = (time.perf_counter() - start) * 1000
                
                # Use std to determine if noise was generated
                if std == 0:
                    noise_stats['noise_generation_zero'].append(duration)
                else:
                    noise_stats['noise_generation_nonzero'].append(duration)
                
                return result
                
            wrapper.noise_stats = noise_stats
            return wrapper
                    
        # Patch opacus _noise_generation
        self.original_generate_noise = opacus_opt._generate_noise
        self.wrapped_noise_gen = timed_noise_generation(self.original_generate_noise)
        opacus_opt._generate_noise = self.wrapped_noise_gen
    
        # Create base model based on config type
        if config.model_type == "resnet":
            self.model = create_dp_resnet18().to(config.device)
        else:  # default to transformer
            self.model = DPTransformerEncoder(config).to(config.device)
        
        # Validate and fix model for Opacus compatibility
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)
            
        # Setup optimizer with a low learning rate for stability
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Initialize privacy engine with a dummy data loader
        print("Initializing privacy engine...")
        self.privacy_engine = PrivacyEngine()
        
        # Create dummy data loader with correct batch size
        dummy_data = torch.utils.data.DataLoader(
            [(torch.randn(self.config.sequence_length, self.config.hidden_dim), 
              torch.tensor(0)) for _ in range(100)],
            batch_size=self.config.batch_size
        )
        
        # Create the dataset and loader first
        print("Creating dummy dataset...")
        dummy_dataset = self.create_dummy_dataset()
        self.train_loader = torch.utils.data.DataLoader(
            dummy_dataset,
            batch_size=self.config.batch_size,
        )

        # Make DP model and optimizer
        try:
            print("Setting up private model and optimizer...")
            self.dp_model, self.dp_optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,  # Use the loader we just created
                noise_multiplier=config.noise_multiplier,
                max_grad_norm=config.max_grad_norm,
                poisson_sampling=config.poisson_sampling  # Now this should work with BatchMemoryManager
            )
            print("Private model setup complete.")
        except Exception as e:
            print(f"Error during privacy engine setup: {e}")
            raise
            
    def __del__(self):
        # Restore original function when done
        opacus_opt._generate_noise = self.original_generate_noise    
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage"""
        stats = {
            'cpu_memory': psutil.Process().memory_info().rss / 1024**2  # MB
        }
        
        if self.config.device == "mps":
            # Note: These are not exact equivalents to CUDA memory stats
            stats['gpu_current'] = torch.mps.current_allocated_memory() / 1024**2
            stats['gpu_driver'] = torch.mps.driver_allocated_memory() / 1024**2
            
        return stats
    
    def create_dummy_dataset(self, size: int = 50000) -> torch.utils.data.Dataset:
        """Create a more realistic sized dummy dataset"""
        if self.config.model_type == "resnet":
            data = torch.randn(
                size,
                self.config.in_channels,
                self.config.image_size,
                self.config.image_size,
                device=self.config.device  # Add device here
            )
        else:  # transformer
            data = torch.randn(
                size,
                self.config.sequence_length,
                self.config.hidden_dim,
                device=self.config.device  # Add device here
            )
        
        labels = torch.randint(0, 2, (size,), device=self.config.device)  # Add device here
        return torch.utils.data.TensorDataset(data, labels)

    # And update _generate_dummy_batch to handle both model types:
    def _generate_dummy_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dummy data for benchmarking based on model type"""
        if self.config.model_type == "resnet":
            x = torch.randn(
                self.config.batch_size, 
                self.config.in_channels,
                self.config.image_size,
                self.config.image_size,
                device=self.config.device
            )
        else:  # transformer
            x = torch.randn(
                self.config.batch_size, 
                self.config.sequence_length, 
                self.config.hidden_dim, 
                device=self.config.device
            )
        
        y = torch.randint(
            0, 2, (self.config.batch_size,), 
            device=self.config.device
        )
        return x, y

    def _benchmark_step(
        self,
        variant: str,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """Run and time a single training step with noise debugging"""
        # Initialize class variable for first run tracking if it doesn't exist
        if not hasattr(self, '_first_runs'):
            self._first_runs = {'dpsgd': True, 'dpsgd_no_noise': True, 'vanilla': True}
            
        # Use privacy engine's built-in noise control
        if variant == "dpsgd_no_noise":
            self.privacy_engine.noise_multiplier = 0.0
            if hasattr(self.dp_optimizer, 'noise_multiplier'):
                self.dp_optimizer.noise_multiplier = 0.0
        else:
            self.privacy_engine.noise_multiplier = self.config.noise_multiplier
            if hasattr(self.dp_optimizer, 'noise_multiplier'):
                self.dp_optimizer.noise_multiplier = self.config.noise_multiplier

        # Log first run settings        
        if self._first_runs.get(variant, True):
            print(f"\n{variant.upper()} Settings:")
            print(f"Privacy Engine noise_multiplier: {self.privacy_engine.noise_multiplier}")
            print(f"Optimizer noise_multiplier: {getattr(self.dp_optimizer, 'noise_multiplier', 'Not found')}")
            self._first_runs[variant] = False
        
        # Synchronize before starting step timer
        self.synchronize_device(self.config.device)
        start_time = time.perf_counter()
        
        model = self.dp_model
        optimizer = self.dp_optimizer
                
        with timer("forward_backward", self.timing_stats, self.config.device):
            predictions = model(data)
            loss = self.criterion(predictions, labels).mean()
            optimizer.zero_grad()
            loss.backward()
            
        with timer("optimizer_step", self.timing_stats, self.config.device):
            optimizer.step()
        
        self.synchronize_device(self.config.device)
        step_time = time.perf_counter() - start_time
        memory_stats = self._get_memory_stats()
            
        return step_time, memory_stats
    
    def run_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run full benchmark suite"""
        variants = ["dpsgd", "dpsgd_no_noise"]  # Remove vanilla SGD
        results = {v: {
            'time': [], 
            'cpu_memory': [], 
            'gpu_current': [],
            'gpu_driver': [],
            'max_cpu_memory': 0,
            'max_gpu_current': 0,
            'max_gpu_driver': 0
        } for v in variants}
        # Warmup
        print("\nWarmup...")
        for _ in range(self.config.warmup_iterations):
            for variant in variants:
                with BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=self.config.batch_size,
                    optimizer=self.dp_optimizer
                ) as memory_safe_data_loader:
                    for batch in memory_safe_data_loader:
                        data, labels = batch
                        self._benchmark_step(variant, data, labels)
                        break  # Only one batch for warmup
                        
        # Reset timing stats after warmup
        self.timing_stats.clear()
        self.wrapped_noise_gen.noise_stats['noise_generation_nonzero'].clear()
        self.wrapped_noise_gen.noise_stats['noise_generation_zero'].clear()
        
        # Main benchmark loop
        print("\nRunning main benchmark...")
        for i in range(self.config.num_iterations):
            if i % 10 == 0:
                print(f"Iteration {i}/{self.config.num_iterations}")
            
            for variant in variants:
                with BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=self.config.batch_size,
                    optimizer=self.dp_optimizer
                ) as memory_safe_data_loader:
                    # Time the data loading/sampling specifically
                    with timer("data_sampling", self.timing_stats, self.config.device):
                        for batch in memory_safe_data_loader:
                            data, labels = batch
                            break  # Get timing for just the first batch
                    for batch in memory_safe_data_loader:
                        data, labels = batch
                        step_time, memory_stats = self._benchmark_step(variant, data, labels)
                        break  # Only process one batch per iteration
                
                # Record results
                results[variant]['time'].append(step_time)
                results[variant]['cpu_memory'].append(memory_stats['cpu_memory'])
                
                if self.config.device == "mps":
                    results[variant]['gpu_current'].append(memory_stats['gpu_current'])
                    results[variant]['gpu_driver'].append(memory_stats['gpu_driver'])
                    
                    # Track peak memory
                    results[variant]['max_cpu_memory'] = max(
                        results[variant]['max_cpu_memory'],
                        memory_stats['cpu_memory']
                    )
                    results[variant]['max_gpu_current'] = max(
                        results[variant]['max_gpu_current'],
                        memory_stats['gpu_current']
                    )
                    results[variant]['max_gpu_driver'] = max(
                        results[variant]['max_gpu_driver'],
                        memory_stats['gpu_driver']
                    )
        
        # Calculate statistics (same as before)
        summary = {}
        for variant in variants:
            summary[variant] = {
                'avg_time': np.mean(results[variant]['time']),
                'std_time': np.std(results[variant]['time']),
                'avg_cpu_memory': np.mean(results[variant]['cpu_memory']),
                'avg_gpu_current': np.mean(results[variant]['gpu_current']) if self.config.device == "mps" else 0,
                'avg_gpu_driver': np.mean(results[variant]['gpu_driver']) if self.config.device == "mps" else 0,
                'max_cpu_memory': results[variant]['max_cpu_memory'],
                'max_gpu_current': results[variant]['max_gpu_current'],
                'max_gpu_driver': results[variant]['max_gpu_driver'],
                'throughput': self.config.batch_size / np.mean(results[variant]['time'])
            }
                
        return summary

    def print_timing_statistics(self):
        """Print detailed timing statistics including separated noise generation times"""
        # Regular timing stats
        for operation, times in self.timing_stats.items():
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            print(f"{operation:20s}: {mean_time:8.2f} ± {std_time:6.2f}")
            
        # Separated noise generation stats
        for key, times in self.wrapped_noise_gen.noise_stats.items():
            if times:  # Only print if we have measurements
                mean_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                print(f"{key:20s}: {mean_time:8.4f} ± {std_time:6.4f}")

def main(model_type = "resnet", poisson_sampling = True):
    # Run one model at a time based on command line argument or config
    config = BenchmarkConfig(model_type=model_type, poisson_sampling=poisson_sampling)
    
    print(f"\nRunning {config.model_type} benchmark {'WITH' if config.poisson_sampling else 'WITHOUT'} poisson subsampling")
    benchmark = DPSGDBenchmark(config)
    results = benchmark.run_benchmark()
    
    # Add clear indicator of sampling mode
    print("\n" + "="*50)
    print(f"SAMPLING MODE: {'POISSON' if config.poisson_sampling else 'STANDARD'}")
    print("="*50 + "\n")
    
    # Print results with detailed memory information
    print("\nBenchmark Results:")
    print("-" * 80)
    for variant, metrics in results.items():
        print(f"\n{variant.upper()}:")
        print(f"Average step time: {metrics['avg_time']*1000:.2f} ms (± {metrics['std_time']*1000:.2f} ms)")
        print(f"Throughput: {metrics['throughput']:.2f} examples/sec")
        
        # Memory statistics
        print("\nMemory Usage:")
        print(f"CPU Memory: {metrics['avg_cpu_memory']:.2f} MB")
        if 'avg_gpu_current' in metrics and metrics['avg_gpu_current'] > 0:
            print(f"GPU Current Memory: {metrics['avg_gpu_current']:.2f} MB")
            print(f"GPU Driver Memory: {metrics['avg_gpu_driver']:.2f} MB")
            
        # Memory peak differences
        if 'max_cpu_memory' in metrics:
            print(f"\nPeak Memory Usage:")
            print(f"Peak CPU Memory: {metrics['max_cpu_memory']:.2f} MB")
            if 'max_gpu_current' in metrics and metrics['max_gpu_current'] > 0:
                print(f"Peak GPU Current Memory: {metrics['max_gpu_current']:.2f} MB")
                print(f"Peak GPU Driver Memory: {metrics['max_gpu_driver']:.2f} MB")
    
    # Print detailed timing statistics
    print("\nDetailed Timing Statistics:")
    print("-" * 80)
    benchmark.print_timing_statistics()

if __name__ == "__main__":
    main()
