import numpy as np
import tensorflow as tf
import torch
import time
import psutil
import os
from memory_profiler import profile

class RNGBenchmark:
    def __init__(self, size=1000000):
        self.size = size
        self.process = psutil.Process(os.getpid())
        
        # Check for MPS availability and initialize
        self.mps_available = torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_device = torch.device("mps")
            print("MPS (Metal Performance Shaders) is available")
            # Get initial GPU memory info
            self._print_gpu_memory_info()
        else:
            print("MPS is not available, running PyTorch on CPU only")
            
    def _print_gpu_memory_info(self):
        """Print current GPU memory usage"""
        if hasattr(torch.mps, 'current_allocated_memory'):
            print(f"\nGPU Memory Info:")
            print(f"Current allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
            print(f"Driver allocated: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
            # Note: On Apple Silicon, available memory is dynamic and shared with systemf
            
    def _measure_memory(self):
        """Return memory usage dictionary with CPU and GPU metrics"""
        memory_info = {
            'cpu_memory': self.process.memory_info().rss / 1024**2,  # MB
            'gpu_memory': 0
        }
        
        if self.mps_available:
            memory_info['gpu_memory'] = torch.mps.current_allocated_memory() / 1024**2  # MB
            
        return memory_info
            
    @profile
    def benchmark_numpy(self):
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers
        _ = np.random.random(self.size)
        
        elapsed = time.time() - start_time
        end_mem = self._measure_memory()
        
        return {
            'method': 'numpy',
            'device': 'cpu',
            'time': elapsed,
            'cpu_memory_used': end_mem['cpu_memory'] - start_mem['cpu_memory'],
            'gpu_memory_used': end_mem['gpu_memory'] - start_mem['gpu_memory'],
            'throughput': self.size / elapsed / 1e6
        }
    
    @profile
    def benchmark_pytorch_cpu(self):
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers on CPU
        _ = torch.rand(self.size)
        
        elapsed = time.time() - start_time
        end_mem = self._measure_memory()
        
        return {
            'method': 'pytorch',
            'device': 'cpu',
            'time': elapsed,
            'cpu_memory_used': end_mem['cpu_memory'] - start_mem['cpu_memory'],
            'gpu_memory_used': end_mem['gpu_memory'] - start_mem['gpu_memory'],
            'throughput': self.size / elapsed / 1e6
        }
    
    @profile
    def benchmark_pytorch_mps(self):
        if not self.mps_available:
            return None
            
        # Clear GPU memory cache
        torch.mps.empty_cache()
        
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers on MPS
        _ = torch.rand(self.size, device=self.mps_device)
        # Ensure computation is complete
        torch.mps.synchronize()
        
        elapsed = time.time() - start_time
        end_mem = self._measure_memory()
        
        return {
            'method': 'pytorch',
            'device': 'mps',
            'time': elapsed,
            'cpu_memory_used': end_mem['cpu_memory'] - start_mem['cpu_memory'],
            'gpu_memory_used': end_mem['gpu_memory'] - start_mem['gpu_memory'],
            'throughput': self.size / elapsed / 1e6
        }
    
    @profile
    def benchmark_tensorflow(self):
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers
        _ = tf.random.uniform((self.size,))
        
        elapsed = time.time() - start_time
        end_mem = self._measure_memory()
        
        return {
            'method': 'tensorflow',
            'device': 'metal',  # TF automatically uses Metal on Apple Silicon
            'time': elapsed,
            'cpu_memory_used': end_mem['cpu_memory'] - start_mem['cpu_memory'],
            'gpu_memory_used': end_mem['gpu_memory'] - start_mem['gpu_memory'],
            'throughput': self.size / elapsed / 1e6
        }

    def run_all_benchmarks(self, iterations=5):
        results = []
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            self._print_gpu_memory_info()
            results.extend([
                self.benchmark_numpy(),
                self.benchmark_pytorch_cpu(),
                self.benchmark_pytorch_mps(),
                self.benchmark_tensorflow()
            ])
            # Clear GPU memory between iterations
            if self.mps_available:
                torch.mps.empty_cache()
        return results

if __name__ == "__main__":
    benchmark = RNGBenchmark(size=10_000_000)
    results = benchmark.run_all_benchmarks()
    
    # Print results
    print("\nRNG Performance Benchmark Results:")
    print("-" * 120)
    print(f"{'Method':<12} {'Device':<8} {'Avg Time (s)':<12} {'CPU Mem (MB)':<15} {'GPU Mem (MB)':<15} {'Throughput (M/s)':<15}")
    print("-" * 120)
    
    methods = [('numpy', 'cpu'), ('pytorch', 'cpu'), ('pytorch', 'mps'), ('tensorflow', 'metal')]
    for method, device in methods:
        method_results = [r for r in results if r and r['method'] == method and r['device'] == device]
        if method_results:
            avg_time = sum(r['time'] for r in method_results) / len(method_results)
            avg_cpu_mem = sum(r['cpu_memory_used'] for r in method_results) / len(method_results)
            avg_gpu_mem = sum(r['gpu_memory_used'] for r in method_results) / len(method_results)
            avg_throughput = sum(r['throughput'] for r in method_results) / len(method_results)
            
            print(f"{method:<12} {device:<8} {avg_time:<12.3f} {avg_cpu_mem:<15.2f} {avg_gpu_mem:<15.2f} {avg_throughput:<15.2f}")
