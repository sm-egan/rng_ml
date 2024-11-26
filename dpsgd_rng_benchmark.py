import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

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
    
    def _compute_per_example_grads(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-example gradients"""
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
    
    def _clip_gradients(
        self,
        per_example_grads: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Clip per-example gradients by norm"""
        clipped_grads = {}
        
        for idx in range(self.config.batch_size):
            example_grads = per_example_grads[idx]
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(g) 
                    for g in example_grads.values()
                ])
            )
            scaling_factor = min(1.0, self.config.max_grad_norm / total_norm)
            
            clipped_grads[idx] = {
                k: g * scaling_factor
                for k, g in example_grads.items()
            }
            
        return clipped_grads
    
    def _aggregate_grads(
        self,
        per_example_grads: Dict[str, torch.Tensor],
        add_noise: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Aggregate clipped (and optionally noised) gradients"""
        aggregated = {}
        
        for param_name in per_example_grads[0].keys():
            sum_grads = torch.stack([
                per_example_grads[idx][param_name]
                for idx in range(self.config.batch_size)
            ]).sum(0)
            
            mean_grad = sum_grads / self.config.batch_size
            
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
        profiler: Optional[torch.profiler.profile] = None
    ) -> Tuple[float, float, float]:
        """Run and time a single training step"""
        start_time = time.perf_counter()
        
        if variant == "dpsgd":
            # Full DP-SGD with noise
            with record_function("compute_grads"):
                per_example_grads = self._compute_per_example_grads(data, labels)
            
            with record_function("clip_grads"):
                clipped_grads = self._clip_gradients(per_example_grads)
            
            with record_function("noise_and_aggregate"):
                final_grads = self._aggregate_grads(clipped_grads, add_noise=True)
            
        elif variant == "dpsgd_no_noise":
            with record_function("compute_grads"):
                per_example_grads = self._compute_per_example_grads(data, labels)
            
            with record_function("clip_grads"):
                clipped_grads = self._clip_gradients(per_example_grads)
            
            with record_function("aggregate"):
                final_grads = self._aggregate_grads(clipped_grads, add_noise=False)
            
        else:  # vanilla SGD
            with record_function("standard_sgd"):
                outputs = self.model(data)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                final_grads = {
                    name: param.grad
                    for name, param in self.model.named_parameters()
                }
            
        step_time = time.perf_counter() - start_time
        
        cpu_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        gpu_memory = (
            torch.cuda.max_memory_allocated() / 1024**2  # MB
            if torch.cuda.is_available() 
            else 0
        )
            
        return step_time, cpu_memory, gpu_memory
            
    def run_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run full benchmark suite"""
        variants = ["dpsgd", "dpsgd_no_noise", "vanilla"]
        results = {v: {'time': [], 'cpu_memory': [], 'gpu_memory': []} for v in variants}
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            data, labels = self._generate_dummy_batch()
            for variant in variants:
                self._benchmark_step(variant, data, labels)
                
        # Main benchmark loop
        for i in range(self.config.num_iterations):
            data, labels = self._generate_dummy_batch()
            
            for variant in variants:
                step_time, cpu_mem, gpu_mem = self._benchmark_step(
                    variant, data, labels
                )
                
                results[variant]['time'].append(step_time)
                results[variant]['cpu_memory'].append(cpu_mem)
                results[variant]['gpu_memory'].append(gpu_mem)
                
        # Calculate statistics
        summary = {}
        for variant in variants:
            summary[variant] = {
                'avg_time': np.mean(results[variant]['time']),
                'std_time': np.std(results[variant]['time']),
                'avg_cpu_memory': np.mean(results[variant]['cpu_memory']),
                'avg_gpu_memory': np.mean(results[variant]['gpu_memory']),
                'throughput': self.config.batch_size / np.mean(results[variant]['time'])
            }
            
        return summary

def main():
    # Initialize config with larger model for better profiling
    config = BenchmarkConfig(
        batch_size=64,
        sequence_length=128,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        num_iterations=50,
        warmup_iterations=10
    )
    
    benchmark = DPSGDBenchmark(config)
    
    # Run with profiler
    print("Running benchmark with profiler...")
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        # Run a few iterations with profiler
        for i in range(5):
            data, labels = benchmark._generate_dummy_batch()
            for variant in ["dpsgd", "dpsgd_no_noise", "vanilla"]:
                benchmark._benchmark_step(variant, data, labels, prof)
    
    # Print profiler results
    print("\nProfiler Results:")
    print("-" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))
    
    # Run full benchmark
    print("\nRunning full benchmark...")
    results = benchmark.run_benchmark()
    
    # Print benchmark results
    print("\nBenchmark Results:")
    print("-" * 80)
    for variant, metrics in results.items():
        print(f"\n{variant.upper()}:")
        print(f"Average step time: {metrics['avg_time']*1000:.2f} ms (± {metrics['std_time']*1000:.2f} ms)")
        print(f"Throughput: {metrics['throughput']:.2f} examples/sec")
        print(f"CPU Memory: {metrics['avg_cpu_memory']:.2f} MB")
        print(f"GPU Memory: {metrics['avg_gpu_memory']:.2f} MB")
    
    if 'dpsgd' in results and 'vanilla' in results:
        overhead = results['dpsgd']['avg_time'] - results['vanilla']['avg_time']
        print(f"\nTotal DP-SGD overhead: {overhead*1000:.2f} ms per step")
        
    if 'dpsgd' in results and 'dpsgd_no_noise' in results:
        rng_overhead = results['dpsgd']['avg_time'] - results['dpsgd_no_noise']['avg_time']
        print(f"RNG overhead: {rng_overhead*1000:.2f} ms per step")
        rng_percent = (rng_overhead / results['dpsgd']['avg_time']) * 100
        print(f"RNG overhead percentage: {rng_percent:.1f}% of total DP-SGD time")

if __name__ == "__main__":
    main()