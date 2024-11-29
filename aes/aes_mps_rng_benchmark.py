import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from aes_rng import AESRandomGenerator

class MPSAESRandomGenerator(AESRandomGenerator):
    """Extended AES Generator with MPS support and float32 handling"""
    def rand(self, *size: int) -> torch.Tensor:
        """Generate uniform random float tensor, converting to float32 for MPS"""
        n_values = np.prod(size)
        random_bytes = self._generate_random_bytes(n_values * 4)
        random_array = self._bytes_to_float_array(random_bytes, size)
        # Convert to float32 before moving to MPS
        return torch.from_numpy(random_array).to(dtype=torch.float32).to(self.device)
    
    def randn(self, *size: int) -> torch.Tensor:
        """Generate normal distribution tensor, ensuring float32 for MPS"""
        u1 = self.rand(*size)
        u2 = self.rand(*size)
        
        angle = 2 * np.pi * u1
        radius = torch.sqrt(-2 * torch.log(u2))
        return radius * torch.cos(angle)

def benchmark_rngs(sizes: List[int], use_mps: bool = False, repeat: int = 10) -> Dict:
    """Benchmark different RNG implementations across various tensor sizes."""
    results = {
        'sizes': sizes,
        'pytorch': {'mean': [], 'std': []},
        'aes': {'mean': [], 'std': []}
    }
    
    device = torch.device("mps") if use_mps else torch.device("cpu")
    print(f"Running benchmark on device: {device}")
    
    # Initialize generators
    torch_gen = torch.Generator(device='cpu')  # MPS doesn't support generator yet
    aes_gen = MPSAESRandomGenerator(device=device) if use_mps else AESRandomGenerator()
    
    for size in sizes:
        print(f"Benchmarking size {size}...")
        
        # PyTorch default RNG
        def torch_op():
            # Explicitly use float32 for MPS
            if use_mps:
                return torch.rand(size, dtype=torch.float32, generator=torch_gen).to(device)
            return torch.rand(size, generator=torch_gen)
            
        mean, std = time_operation(torch_op, repeat)
        results['pytorch']['mean'].append(mean)
        results['pytorch']['std'].append(std)
        
        # AES RNG
        mean, std = time_operation(
            lambda: aes_gen.rand(size),
            repeat
        )
        results['aes']['mean'].append(mean)
        results['aes']['std'].append(std)
        
        # Force MPS synchronization if needed
        if use_mps:
            torch.mps.synchronize()
    
    return results

def time_operation(func, repeat=10):
    """Time an operation multiple times and return mean and std in milliseconds"""
    times = []
    for _ in range(repeat):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        start = time.perf_counter()
        func()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    return np.mean(times), np.std(times)

def plot_results(results: Dict, device: str):
    """Plot benchmark results with error bars"""
    plt.figure(figsize=(10, 6))
    
    sizes = np.array(results['sizes'])
    
    plt.errorbar(sizes, results['pytorch']['mean'], 
                yerr=results['pytorch']['std'],
                label=f'PyTorch RNG ({device})', marker='o')
    
    plt.errorbar(sizes, results['aes']['mean'], 
                yerr=results['aes']['std'],
                label=f'AES RNG ({device})', marker='s')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tensor Size')
    plt.ylabel('Time (ms)')
    plt.title(f'RNG Performance Comparison on {device}')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    # Test sizes (powers of 10)
    sizes = [10**i for i in range(1, 8)]  # 10 to 10,000,000
    
    # Check if MPS is available
    use_mps = torch.backends.mps.is_available()
    device_name = "MPS" if use_mps else "CPU"
    
    # Run benchmarks
    print(f"Starting benchmarks on {device_name}...")
    results = benchmark_rngs(sizes, use_mps=use_mps)
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Size':<10} {'PyTorch (ms)':<20} {'AES (ms)':<20}")
    print("-" * 50)
    for i, size in enumerate(results['sizes']):
        print(f"{size:<10} "
              f"{results['pytorch']['mean'][i]:>8.3f} ± {results['pytorch']['std'][i]:<8.3f} "
              f"{results['aes']['mean'][i]:>8.3f} ± {results['aes']['std'][i]:<8.3f}")
    
    # Plot results
    plot_results(results, device_name)
