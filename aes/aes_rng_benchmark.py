import torch
import time
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Import the AES RNG implementation
from aes_rng import AESRandomGenerator  # Save the previous code as aes_rng.py

def time_operation(func, repeat=10):
    """Time an operation multiple times and return mean and std in milliseconds"""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    return np.mean(times), np.std(times)

def benchmark_rngs(sizes: List[int], repeat: int = 10) -> Dict:
    """
    Benchmark different RNG implementations across various tensor sizes.
    Returns timing results in milliseconds.
    """
    results = {
        'sizes': sizes,
        'pytorch': {'mean': [], 'std': []},
        'aes': {'mean': [], 'std': []}
    }
    
    # Initialize generators
    torch_gen = torch.Generator()
    aes_gen = AESRandomGenerator()
    
    for size in sizes:
        print(f"Benchmarking size {size}...")
        
        # PyTorch default RNG
        mean, std = time_operation(
            lambda: torch.randn(size, generator=torch_gen),
            repeat
        )
        results['pytorch']['mean'].append(mean)
        results['pytorch']['std'].append(std)
        
        # AES RNG
        mean, std = time_operation(
            lambda: aes_gen.randn(size),
            repeat
        )
        results['aes']['mean'].append(mean)
        results['aes']['std'].append(std)
    
    return results

def plot_results(results: Dict):
    """Plot benchmark results with error bars"""
    plt.figure(figsize=(10, 6))
    
    sizes = np.array(results['sizes'])
    
    # Plot PyTorch results
    plt.errorbar(sizes, results['pytorch']['mean'], 
                yerr=results['pytorch']['std'],
                label='PyTorch RNG', marker='o')
    
    # Plot AES results
    plt.errorbar(sizes, results['aes']['mean'], 
                yerr=results['aes']['std'],
                label='AES RNG', marker='s')
    
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
    sizes = [10**i for i in range(1, 8)]  # 10 to 1,000,000
    
    # Run benchmarks
    print("Starting benchmarks...")
    results = benchmark_rngs(sizes)
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Size':<10} {'PyTorch (ms)':<20} {'AES (ms)':<20}")
    print("-" * 60)
    for i, size in enumerate(results['sizes']):
        print(f"{size:<10} "
              f"{results['pytorch']['mean'][i]:>8.3f} ± {results['pytorch']['std'][i]:<8.3f} "
              f"{results['aes']['mean'][i]:>8.3f} ± {results['aes']['std'][i]:<8.3f}")
    
    # Plot results
    plot_results(results)
