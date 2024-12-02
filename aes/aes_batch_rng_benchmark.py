import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from aes_rng import AESRandomGenerator
from batched_aes_rng import BatchedAESRandomGenerator  # Save the previous code as batched_aes_rng.py

def time_operation(func, repeat=10):
    """Time an operation multiple times and return mean and std in milliseconds"""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    return np.mean(times), np.std(times)

def benchmark_rngs(sizes: List[int], repeat: int = 10) -> Dict:
    """Benchmark different RNG implementations across various tensor sizes."""
    results = {
        'sizes': sizes,
        'pytorch': {'mean': [], 'std': []},
        'aes': {'mean': [], 'std': []},
        'batched_aes': {'mean': [], 'std': []}
    }
    
    # Initialize generators
    torch_gen = torch.Generator()
    aes_gen = AESRandomGenerator()
    batched_gen = BatchedAESRandomGenerator(batch_size=10_000_000)
    
    for size in sizes:
        print(f"Benchmarking size {size}...")
        
        # PyTorch default RNG
        mean, std = time_operation(
            lambda: torch.randn(size, generator=torch_gen),
            repeat
        )
        results['pytorch']['mean'].append(mean)
        results['pytorch']['std'].append(std)
        
        # Original AES RNG
        mean, std = time_operation(
            lambda: aes_gen.randn(size),
            repeat
        )
        results['aes']['mean'].append(mean)
        results['aes']['std'].append(std)
        
        # Batched AES RNG
        mean, std = time_operation(
            lambda: batched_gen.rand(size),
            repeat
        )
        results['batched_aes']['mean'].append(mean)
        results['batched_aes']['std'].append(std)
    
    return results

def plot_results(results: Dict):
    """Plot benchmark results with error bars"""
    plt.figure(figsize=(10, 6))
    
    sizes = np.array(results['sizes'])
    
    plt.errorbar(sizes, results['pytorch']['mean'], 
                yerr=results['pytorch']['std'],
                label='PyTorch RNG', marker='o')
    
    plt.errorbar(sizes, results['aes']['mean'], 
                yerr=results['aes']['std'],
                label='AES RNG', marker='s')
    
    plt.errorbar(sizes, results['batched_aes']['mean'], 
                yerr=results['batched_aes']['std'],
                label='Batched AES RNG', marker='^')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tensor Size')
    plt.ylabel('Time (ms)')
    plt.title('RNG Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    # Test sizes (powers of 10)
    sizes = [10**i for i in range(1, 8)]  # 10 to 10,000,000
    
    # Run benchmarks
    print("Starting benchmarks...")
    results = benchmark_rngs(sizes)
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 70)
    print(f"{'Size':<10} {'PyTorch (ms)':<20} {'AES (ms)':<20} {'Batched AES (ms)':<20}")
    print("-" * 70)
    for i, size in enumerate(results['sizes']):
        print(f"{size:<10} "
              f"{results['pytorch']['mean'][i]:>8.3f} ± {results['pytorch']['std'][i]:<8.3f} "
              f"{results['aes']['mean'][i]:>8.3f} ± {results['aes']['std'][i]:<8.3f} "
              f"{results['batched_aes']['mean'][i]:>8.3f} ± {results['batched_aes']['std'][i]:<8.3f}")
    
    # Plot results
    plot_results(results)
