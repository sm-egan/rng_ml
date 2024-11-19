import numpy as np
import tensorflow as tf
import torch
import time
from collections import defaultdict

class RNGBenchmarkV4:
    def __init__(self, sizes=[1000000, 10000000, 100000000]):
        self.sizes = sizes
        self.warmup_rounds = 5
        
        # PyTorch MPS setup
        self.mps_available = torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_device = torch.device("mps")
            print(f"PyTorch MPS device: {self.mps_device}")
        
        # TensorFlow GPU setup
        physical_devices = tf.config.list_physical_devices()
        print("Available TF devices:", physical_devices)
        
        # Try to configure TensorFlow for GPU/Metal
        try:
            tf.config.experimental.set_memory_growth('GPU', True)
            print("TensorFlow GPU memory growth enabled")
        except:
            print("Could not configure TensorFlow GPU memory growth")
            
    def measure_transfer_time(self, size):
        """Measure time to transfer data to and from GPU"""
        if not self.mps_available:
            return None
            
        # CPU to GPU transfer
        cpu_tensor = torch.rand(size)
        start_time = time.time()
        gpu_tensor = cpu_tensor.to(self.mps_device)
        torch.mps.synchronize()
        to_gpu_time = time.time() - start_time
        
        # GPU to CPU transfer
        start_time = time.time()
        _ = gpu_tensor.cpu()
        torch.mps.synchronize()
        to_cpu_time = time.time() - start_time
        
        return {
            'size': size,
            'to_gpu_time': to_gpu_time,
            'to_cpu_time': to_cpu_time,
            'total_transfer_time': to_gpu_time + to_cpu_time
        }
            
    def benchmark_pytorch_mps(self, size):
        if not self.mps_available:
            return None
            
        torch.mps.empty_cache()
        
        # Measure transfer overhead
        transfer_times = self.measure_transfer_time(size)
        
        # Generation timing with events
        start_event = torch.mps.Event(enable_timing=True)
        end_event = torch.mps.Event(enable_timing=True)
        
        start_event.record()
        tensor = torch.rand(size, device=self.mps_device)
        end_event.record()
        
        # Wait for computation and get GPU time
        torch.mps.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to s
        
        return {
            'method': 'pytorch',
            'device': 'mps',
            'size': size,
            'compute_time': gpu_time,
            'transfer_times': transfer_times,
            'total_time': gpu_time + transfer_times['total_transfer_time'],
            'compute_throughput': size / gpu_time / 1e6,
            'effective_throughput': size / (gpu_time + transfer_times['total_transfer_time']) / 1e6
        }
        
    def benchmark_tensorflow(self, size):
        # Force TensorFlow to use GPU if available
        with tf.device('/GPU:0'):
            # Time generation
            start_time = time.time()
            tensor = tf.random.uniform((size,))
            
            # Force execution completion
            tensor = tensor.numpy()  # This will transfer back to CPU
            elapsed = time.time() - start_time
            
            return {
                'method': 'tensorflow',
                'device': 'gpu',
                'size': size,
                'time': elapsed,
                'throughput': size / elapsed / 1e6
            }

    def run_benchmarks(self, iterations=3):
        results = defaultdict(list)
        
        for size in self.sizes:
            print(f"\nBenchmarking with size: {size:,}")
            for i in range(iterations):
                print(f"\nIteration {i+1}/{iterations}")
                
                if self.mps_available:
                    result = self.benchmark_pytorch_mps(size)
                    if result:
                        results[size].append(result)
                        print(f"PyTorch MPS - Size: {size:,}")
                        print(f"  Compute time: {result['compute_time']:.3f}s")
                        print(f"  Transfer to GPU: {result['transfer_times']['to_gpu_time']:.3f}s")
                        print(f"  Transfer to CPU: {result['transfer_times']['to_cpu_time']:.3f}s")
                        print(f"  Compute throughput: {result['compute_throughput']:.2f} M/s")
                        print(f"  Effective throughput: {result['effective_throughput']:.2f} M/s")
                
                result = self.benchmark_tensorflow(size)
                results[size].append(result)
                print(f"TensorFlow - Size: {size:,}")
                print(f"  Total time: {result['time']:.3f}s")
                print(f"  Throughput: {result['throughput']:.2f} M/s")
                
        return results

    def print_results(self, results):
        print("\nRNG Performance Benchmark Results:")
        print("-" * 120)
        print(f"{'Size':<12} {'Method':<12} {'Device':<8} {'Avg Time (s)':<12} {'Throughput (M/s)':<15}")
        print("-" * 120)
        
        for size in self.sizes:
            size_results = results[size]
            by_method = defaultdict(list)
            for r in size_results:
                key = (r['method'], r['device'])
                by_method[key].append(r)
            
            for (method, device), method_results in by_method.items():
                avg_time = sum(r['time'] for r in method_results) / len(method_results)
                avg_throughput = sum(r['throughput'] for r in method_results) / len(method_results)
                print(f"{size:<12,} {method:<12} {device:<8} {avg_time:<12.3f} {avg_throughput:<15.2f}")

if __name__ == "__main__":
    benchmark = RNGBenchmarkV4()
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)