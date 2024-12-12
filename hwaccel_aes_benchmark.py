import time
import torch
import numpy as np
from typing import List, Tuple
import statistics
from aes_prng import (
    BatchedAESRandomGenerator,
    HWAccelBatchedAESRandomGenerator
)

def calculate_throughput(size: Tuple[int, ...], time_ms: float) -> float:
    """Calculate throughput in Gbps given tensor size and time in ms"""
    # Each element is a 32-bit (4 byte) float
    total_bytes = np.prod(size) * 4
    # Convert ms to seconds and bytes to bits
    throughput_gbps = (total_bytes * 8) / (time_ms * 1e-3) / 1e9
    return throughput_gbps

def benchmark_generator(generator, sizes: List[Tuple[int, ...]], n_trials: int = 100, is_pytorch: bool = False):
    """Benchmark a generator's performance on different tensor sizes"""
    results = {}
    
    for size in sizes:
        times = []
        if is_pytorch:
            # For PyTorch, create a tensor once and reuse it
            tensor = torch.empty(size)
        
        # Warmup
        for _ in range(10):
            if is_pytorch:
                tensor.random_()
            else:
                _ = generator.rand(*size)
            
        # Timing runs
        for _ in range(n_trials):
            start = time.perf_counter()
            if is_pytorch:
                tensor.random_()
            else:
                _ = generator.rand(*size)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
        mean_time = statistics.mean(times)
        results[size] = {
            'mean': mean_time,
            'std': statistics.stdev(times),
            'throughput': calculate_throughput(size, mean_time)
        }
    
    return results

def main():
    # Test sizes that match typical parameter tensors
    sizes = [
        (256, 256),      # Embedding layer (~0.25M elements)
        (3, 256, 256),   # Self-attention weights (~0.75M elements)
        (1024, 1024),    # Large parameter tensor (~1M elements)
        (9, 1024, 1024)  # Very large parameter tensor (~10M elements)
    ]
    
    print("Testing PyTorch's default PRNG...")
    pytorch_results = benchmark_generator(None, sizes, is_pytorch=True)
    
    print("\nTesting original AES implementation...")
    orig_gen = BatchedAESRandomGenerator()
    orig_results = benchmark_generator(orig_gen, sizes)
    
    print("\nTesting hardware-accelerated implementation...")
    hw_gen = HWAccelBatchedAESRandomGenerator()
    hw_results = benchmark_generator(hw_gen, sizes)
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    for size in sizes:
        n_elements = np.prod(size)
        print(f"\nTensor size {size} ({n_elements:,} elements):")
        print(f"PyTorch:      {pytorch_results[size]['mean']:6.3f} ± {pytorch_results[size]['std']:6.3f} ms")
        print(f"             {pytorch_results[size]['throughput']:6.1f} Gbps")
        print(f"Original:     {orig_results[size]['mean']:6.3f} ± {orig_results[size]['std']:6.3f} ms")
        print(f"             {orig_results[size]['throughput']:6.1f} Gbps")
        print(f"HW-Accel:     {hw_results[size]['mean']:6.3f} ± {hw_results[size]['std']:6.3f} ms")
        print(f"             {hw_results[size]['throughput']:6.1f} Gbps")
        
        # Calculate speedups relative to PyTorch
        pytorch_time = pytorch_results[size]['mean']
        print("\nSpeedups vs PyTorch:")
        print(f"Original:     {pytorch_time / orig_results[size]['mean']:.2f}x")
        print(f"HW-Accel:     {pytorch_time / hw_results[size]['mean']:.2f}x")

if __name__ == "__main__":
    main()