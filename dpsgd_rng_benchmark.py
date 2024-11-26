import torch
import torch.nn as nn
from torch.func import vmap, grad, functional_call
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import psutil
import numpy as np
from contextlib import contextmanager
import statistics

@contextmanager
def timer(name: str, stats_dict: Dict[str, list]):
    """Custom timer context manager for MPS operations"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    duration_ms = (end - start) * 1000
    if name not in stats_dict:
        stats_dict[name] = []
    stats_dict[name].append(duration_ms)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    batch_size: int = 32
    sequence_length: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    num_iterations: int = 100
    warmup_iterations: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class TransformerEncoder(nn.Module):
    """Simple transformer encoder model for benchmarking"""
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Output projection
        self.classifier = nn.Linear(config.hidden_dim, 2)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)

class DPSGDBenchmark:
    """Benchmark different variants of SGD training"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = TransformerEncoder(config).to(config.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.timing_stats = {}

        # Extract model state for functional transforms
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        
        print(f"Running on device: {config.device}")
        if config.device == "mps":
            print("MPS device properties:")
            print(f"MPS is built: {torch.backends.mps.is_built()}")
            print(f"MPS is available: {torch.backends.mps.is_available()}")
    
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

    def _generate_dummy_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dummy data for benchmarking"""
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
    
    def _compute_loss(self, params, buffers, sample, target):
        """Compute loss for a single example"""
        predictions = functional_call(self.model, (params, buffers), (sample.unsqueeze(0),))
        loss = self.criterion(predictions, target.unsqueeze(0))
        return loss[0]  # Remove batch dimension
        
    def _setup_functorch(self):
        """Setup function transforms for efficient per-example gradients"""
        # Create function to compute gradients for a single example
        compute_grad_single = grad(self._compute_loss)
        
        # Vectorize it over the batch dimension
        # None means don't map over that argument (params and buffers)
        # 0 means map over first dimension of data/targets
        self.compute_grad_batch = vmap(compute_grad_single, in_dims=(None, None, 0, 0))

    def _compute_per_example_grads_naive(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Original naive implementation with separate backward passes"""
        outputs = self.model(data)
        losses = self.criterion(outputs, labels)
        
        per_example_grads = {}
        for loss_idx in range(self.config.batch_size):
            self.model.zero_grad()
            losses[loss_idx].backward(retain_graph=True)
            per_example_grads[loss_idx] = {
                name: param.grad.clone()
                for name, param in self.model.named_parameters()
            }
            
        return per_example_grads

    def _compute_per_example_grads_vectorized(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Optimized implementation using vectorized operations"""
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(data)  # [batch_size, num_classes]
        losses = self.criterion(outputs, labels)  # [batch_size]
        
        # Initialize dict to store gradients
        per_example_grads = {i: {} for i in range(self.config.batch_size)}
        param_names = [name for name, _ in self.model.named_parameters()]
        
        # Compute gradients for the sum of losses to get the Jacobian structure
        total_grad = torch.zeros_like(losses).requires_grad_(True)
        gradients = []
        
        # Compute per-example gradients using implicit Jacobian-vector products
        for i in range(self.config.batch_size):
            # Create mask for this example
            mask = torch.zeros_like(losses)
            mask[i] = 1.0
            
            # Compute gradients for this example
            example_grads = torch.autograd.grad(
                losses,
                self.model.parameters(),
                grad_outputs=mask,
                retain_graph=True,
                allow_unused=True
            )
            
            # Store gradients
            for param_idx, param_name in enumerate(param_names):
                if example_grads[param_idx] is not None:
                    per_example_grads[i][param_name] = example_grads[param_idx].detach()
                
        return per_example_grads
    
    def _compute_per_example_grads_functorch(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-example gradients using functorch"""
        # Compute all gradients in parallel
        grad_values = self.compute_grad_batch(self.params, self.buffers, data, labels)
        
        # Convert to same format as other implementations
        per_example_grads = {i: {} for i in range(self.config.batch_size)}
        
        # Reorganize gradients by example
        for param_name in self.params.keys():
            grads = grad_values[param_name]  # [batch_size, ...]
            for i in range(self.config.batch_size):
                per_example_grads[i][param_name] = grads[i]
                
        return per_example_grads

    def _clip_gradients_vectorized(
        self,
        per_example_grads: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Optimized gradient clipping implementation"""
        clipped_grads = {}
        grad_norms = []
        
        # Compute all gradient norms in parallel
        for idx in range(self.config.batch_size):
            example_grads = list(per_example_grads[idx].values())
            flat_grad = torch.cat([g.flatten() for g in example_grads])
            grad_norms.append(torch.norm(flat_grad))
        
        # Convert to tensor for vectorized operations
        grad_norms = torch.stack(grad_norms)
        scaling_factors = torch.minimum(
            torch.ones_like(grad_norms),
            self.config.max_grad_norm / grad_norms
        )
        
        # Apply scaling factors
        for idx in range(self.config.batch_size):
            clipped_grads[idx] = {
                k: g * scaling_factors[idx]
                for k, g in per_example_grads[idx].items()
            }
            
        return clipped_grads
    
    def _aggregate_grads_vectorized(
        self,
        per_example_grads: Dict[int, Dict[str, torch.Tensor]],
        add_noise: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Optimized gradient aggregation"""
        aggregated = {}
        
        # Get all parameter names from first example
        param_names = list(per_example_grads[0].keys())
        
        for param_name in param_names:
            # Stack gradients for this parameter across all examples
            param_grads = torch.stack([
                per_example_grads[idx][param_name]
                for idx in range(self.config.batch_size)
            ])
            
            # Compute mean
            mean_grad = param_grads.mean(0)
            
            if add_noise:
                noise = torch.randn_like(mean_grad) * \
                    self.config.noise_multiplier * \
                    self.config.max_grad_norm / \
                    self.config.batch_size
                mean_grad += noise
                
            aggregated[param_name] = mean_grad
            
        return aggregated
    
    def _benchmark_step(
        self,
        variant: str,
        data: torch.Tensor,
        labels: torch.Tensor,
        use_optimized: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Run and time a single training step"""
        start_time = time.perf_counter()
        
        compute_grads_fn = (
            self._compute_per_example_grads_vectorized if use_optimized 
            else self._compute_per_example_grads_naive
        )
        clip_grads_fn = (
            self._clip_gradients_vectorized if use_optimized
            else self._clip_gradients
        )
        aggregate_fn = (
            self._aggregate_grads_vectorized if use_optimized
            else self._aggregate_grads
        )
        
        if variant == "dpsgd":
            with timer("compute_grads", self.timing_stats):
                per_example_grads = compute_grads_fn(data, labels)
            
            with timer("clip_grads", self.timing_stats):
                clipped_grads = clip_grads_fn(per_example_grads)
            
            with timer("noise_generation", self.timing_stats):
                final_grads = aggregate_fn(clipped_grads, add_noise=True)
            
        elif variant == "dpsgd_no_noise":
            with timer("compute_grads", self.timing_stats):
                per_example_grads = compute_grads_fn(data, labels)
            
            with timer("clip_grads", self.timing_stats):
                clipped_grads = clip_grads_fn(per_example_grads)
            
            with timer("aggregate_only", self.timing_stats):
                final_grads = aggregate_fn(clipped_grads, add_noise=False)
            
        else:  # vanilla SGD
            with timer("standard_sgd", self.timing_stats):
                outputs = self.model(data)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                final_grads = {
                    name: param.grad
                    for name, param in self.model.named_parameters()
                }
        
        # Ensure MPS is synchronized
        if self.config.device == "mps":
            torch.mps.synchronize()
            
        step_time = time.perf_counter() - start_time
        memory_stats = self._get_memory_stats()
            
        return step_time, memory_stats
            
    def run_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run full benchmark suite"""
        variants = ["dpsgd", "dpsgd_no_noise", "vanilla"]
        results = {v: {
            'time': [], 
            'cpu_memory': [], 
            'gpu_current': [],
            'gpu_driver': []
        } for v in variants}
        
        # Warmup
        print("\nWarmup...")
        for _ in range(self.config.warmup_iterations):
            data, labels = self._generate_dummy_batch()
            for variant in variants:
                self._benchmark_step(variant, data, labels)
                
        # Reset timing stats after warmup
        self.timing_stats.clear()
                
        # Main benchmark loop
        print("\nRunning main benchmark...")
        for i in range(self.config.num_iterations):
            if i % 10 == 0:
                print(f"Iteration {i}/{self.config.num_iterations}")
            
            data, labels = self._generate_dummy_batch()
            
            for variant in variants:
                step_time, memory_stats = self._benchmark_step(variant, data, labels)
                
                results[variant]['time'].append(step_time)
                results[variant]['cpu_memory'].append(memory_stats['cpu_memory'])
                if self.config.device == "mps":
                    results[variant]['gpu_current'].append(memory_stats['gpu_current'])
                    results[variant]['gpu_driver'].append(memory_stats['gpu_driver'])
                
        # Calculate statistics
        summary = {}
        for variant in variants:
            summary[variant] = {
                'avg_time': np.mean(results[variant]['time']),
                'std_time': np.std(results[variant]['time']),
                'avg_cpu_memory': np.mean(results[variant]['cpu_memory']),
                'avg_gpu_current': np.mean(results[variant]['gpu_current']) if self.config.device == "mps" else 0,
                'avg_gpu_driver': np.mean(results[variant]['gpu_driver']) if self.config.device == "mps" else 0,
                'throughput': self.config.batch_size / np.mean(results[variant]['time'])
            }
            
        return summary

    def print_timing_statistics(self):
        """Print detailed timing statistics"""
        print("\nDetailed Timing Statistics (ms):")
        print("-" * 80)
        for operation, times in self.timing_stats.items():
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            print(f"{operation:20s}: {mean_time:8.2f} ± {std_time:6.2f}")

def main():
    config = BenchmarkConfig()
    
    # Run naive implementation first
    print("\nRunning benchmark with naive implementation...")
    benchmark_naive = DPSGDBenchmark(config)
    results_naive = benchmark_naive.run_benchmark()
    
    # Run optimized implementation
    print("\nRunning benchmark with optimized implementation...")
    benchmark_opt = DPSGDBenchmark(config)
    results_opt = benchmark_opt.run_benchmark()
    
    # Print comparison
    print("\nComparison of Implementations:")
    print("-" * 80)
    
    variants = ["dpsgd", "dpsgd_no_noise", "vanilla"]
    for variant in variants:
        naive_time = results_naive[variant]['avg_time'] * 1000
        opt_time = results_opt[variant]['avg_time'] * 1000
        speedup = naive_time / opt_time if opt_time > 0 else 0
        
        print(f"\n{variant.upper()}:")
        print(f"Naive implementation: {naive_time:.2f} ms")
        print(f"Optimized implementation: {opt_time:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
    
    # Print detailed timing statistics for optimized version
    print("\nDetailed Timing Statistics (Optimized Implementation):")
    print("-" * 80)
    benchmark_opt.print_timing_statistics()

if __name__ == "__main__":
    main()