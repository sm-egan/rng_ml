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
        
        # Check for MPS availability
        self.mps_available = torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_device = torch.device("mps")
            print("MPS (Metal Performance Shaders) is available")
        else:
            print("MPS is not available, running PyTorch on CPU only")
            
        # Initialize TensorFlow for Metal
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Metal device found for TensorFlow: {physical_devices}")
            
    def _measure_memory(self):
        """Return memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    @profile
    def benchmark_numpy(self):
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers
        _ = np.random.random(self.size)
        
        elapsed = time.time() - start_time
        mem_used = self._measure_memory() - start_mem
        
        return {
            'method': 'numpy',
            'device': 'cpu',
            'time': elapsed,
            'memory': mem_used,
            'throughput': self.size / elapsed / 1e6
        }
    
    @profile
    def benchmark_pytorch_cpu(self):
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers on CPU
        _ = torch.rand(self.size)
        
        elapsed = time.time() - start_time
        mem_used = self._measure_memory() - start_mem
        
        return {
            'method': 'pytorch',
            'device': 'cpu',
            'time': elapsed,
            'memory': mem_used,
            'throughput': self.size / elapsed / 1e6
        }
    
    @profile
    def benchmark_pytorch_mps(self):
        if not self.mps_available:
            return None
            
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers on MPS
        _ = torch.rand(self.size, device=self.mps_device)
        # Ensure computation is complete
        torch.mps.synchronize()
        
        elapsed = time.time() - start_time
        mem_used = self._measure_memory() - start_mem
        
        return {
            'method': 'pytorch',
            'device': 'mps',
            'time': elapsed,
            'memory': mem_used,
            'throughput': self.size / elapsed / 1e6
        }
    
    @profile
    def benchmark_tensorflow(self):
        start_mem = self._measure_memory()
        start_time = time.time()
        
        # Generate random numbers
        _ = tf.random.uniform((self.size,))
        
        elapsed = time.time() - start_time
        mem_used = self._measure_memory() - start_mem
        
        return {
            'method': 'tensorflow',
            'device': 'metal',  # TF automatically uses Metal on Apple Silicon
            'time': elapsed,
            'memory': mem_used,
            'throughput': self.size / elapsed / 1e6
        }

    def run_all_benchmarks(self, iterations=5):
        results = []
        for _ in range(iterations):
            results.append(self.benchmark_numpy())
            results.append(self.benchmark_pytorch_cpu())
            if self.mps_available:
                mps_result = self.benchmark_pytorch_mps()
                if mps_result:
                    results.append(mps_result)
            results.append(self.benchmark_tensorflow())
        return results

if __name__ == "__main__":
    benchmark = RNGBenchmark(size=10_000_000)
    results = benchmark.run_all_benchmarks()
    
    # Print results
    print("\nRNG Performance Benchmark Results:")
    print("-" * 100)
    print(f"{'Method':<12} {'Device':<8} {'Avg Time (s)':<12} {'Avg Memory (MB)':<15} {'Throughput (M/s)':<15}")
    print("-" * 100)
    
    # Group results by method and device
    from itertools import groupby
    from operator import itemgetter
    
    # Sort results by method and device for grouping
    sorted_results = sorted(results, key=lambda x: (x['method'], x['device']))
    for (method, device), group in groupby(sorted_results, key=lambda x: (x['method'], x['device'])):
        group_list = list(group)
        avg_time = sum(r['time'] for r in group_list) / len(group_list)
        avg_mem = sum(r['memory'] for r in group_list) / len(group_list)
        avg_throughput = sum(r['throughput'] for r in group_list) / len(group_list)
        
        print(f"{method:<12} {device:<8} {avg_time:<12.3f} {avg_mem:<15.2f} {avg_throughput:<15.2f}")
