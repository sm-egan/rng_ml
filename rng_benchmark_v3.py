import numpy as np
import tensorflow as tf
import torch
import time
from collections import defaultdict

class RNGBenchmarkV3:
    def __init__(self, sizes=[1000000, 10000000, 100000000]):
        self.sizes = sizes
        self.warmup_rounds = 5
        
        # Device setup and verification
        self.mps_available = torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_device = torch.device("mps")
            print(f"PyTorch MPS device: {self.mps_device}")
        
        # TensorFlow device verification
        tf_devices = tf.config.list_physical_devices()
        print("TensorFlow devices:", tf_devices)
        
    def warmup(self):
        """Run warmup iterations to ensure GPU is ready"""
        print("\nRunning warmup iterations...")
        warmup_size = 1000000
        for _ in range(self.warmup_rounds):
            if self.mps_available:
                _ = torch.rand(warmup_size, device=self.mps_device)
                torch.mps.synchronize()
            _ = tf.random.uniform((warmup_size,))
            
    def verify_device(self, tensor, expected_device):
        """Verify tensor is on expected device"""
        if isinstance(tensor, torch.Tensor):
            actual_device = tensor.device
        elif isinstance(tensor, tf.Tensor):
            actual_device = tensor.device
        else:
            actual_device = 'cpu'
        print(f"Expected device: {expected_device}, Actual device: {actual_device}")
        
    def benchmark_pytorch_mps(self, size):
        if not self.mps_available:
            return None
            
        torch.mps.empty_cache()
        
        # Start MPS event for more accurate GPU timing
        start_event = torch.mps.Event(enable_timing=True)
        end_event = torch.mps.Event(enable_timing=True)
        
        start_event.record()
        tensor = torch.rand(size, device=self.mps_device)
        end_event.record()
        
        # Verify computation happened on GPU
        self.verify_device(tensor, self.mps_device)
        
        # Wait for computation to finish and get GPU time
        torch.mps.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to s
        
        # Basic statistical verification
        mean = float(tensor.mean().cpu())
        std = float(tensor.std().cpu())
        print(f"PyTorch MPS - Mean: {mean:.3f}, Std: {std:.3f}")
        
        return {
            'method': 'pytorch',
            'device': 'mps',
            'size': size,
            'time': gpu_time,
            'throughput': size / gpu_time / 1e6
        }
        
    def benchmark_tensorflow(self, size):
        # Similar verification for TensorFlow
        start_time = time.time()
        tensor = tf.random.uniform((size,))
        tf.debugging.check_numerics(tensor, "TF tensor validation")
        
        # Force execution of the op
        tensor = tensor.numpy()
        elapsed = time.time() - start_time
        
        mean = float(tf.reduce_mean(tensor))
        std = float(tf.math.reduce_std(tensor))
        print(f"TensorFlow - Mean: {mean:.3f}, Std: {std:.3f}")
        
        return {
            'method': 'tensorflow',
            'device': 'metal',
            'size': size,
            'time': elapsed,
            'throughput': size / elapsed / 1e6
        }

    def run_benchmarks(self, iterations=3):
        self.warmup()
        results = defaultdict(list)
        
        for size in self.sizes:
            print(f"\nBenchmarking with size: {size:,}")
            for i in range(iterations):
                print(f"\nIteration {i+1}/{iterations}")
                
                if self.mps_available:
                    result = self.benchmark_pytorch_mps(size)
                    if result:
                        results[size].append(result)
                
                result = self.benchmark_tensorflow(size)
                results[size].append(result)
                
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
    benchmark = RNGBenchmarkV3()
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)