import numpy as np
import tensorflow as tf
import torch
import time
from collections import defaultdict

class RNGBenchmarkV4:
    def __init__(self, sizes=[np.power(10,n) for n in range(3, 9)]):
        self.sizes = sizes
        self.warmup_rounds = 5
        
        # PyTorch MPS setup
        self.mps_available = torch.backends.mps.is_available()
        if self.mps_available:
            self.mps_device = torch.device("mps")
            print(f"PyTorch MPS device: {self.mps_device}")
        
        # TensorFlow GPU setup
        # physical_devices = tf.config.list_physical_devices()
        # print("Available TF devices:", physical_devices)
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            print("TensorFlow Metal device enabled:", physical_devices[0])
        
        # Try to configure TensorFlow for GPU/Metal
        try:
            tf.config.experimental.set_memory_growth('GPU', True)
            print("TensorFlow GPU memory growth enabled")
        except:
            print("Could not configure TensorFlow GPU memory growth")

    def warmup(self):
        """Run warmup iterations to ensure GPU is ready"""
        print("\nRunning warmup iterations...")
        warmup_size = 1000000
        for _ in range(self.warmup_rounds):
            if self.mps_available:
                _ = torch.rand(warmup_size, device=self.mps_device)
                torch.mps.synchronize()
            _ = tf.random.uniform((warmup_size,))

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
        
    def benchmark_pytorch_cpu(self, size):
        # Generation timing
        start_time = time.time()
        tensor = torch.rand(size)
        elapsed = time.time() - start_time
        
        return {
            'method': 'pytorch',
            'device': 'cpu',
            'size': 'size',
            'total_time': elapsed,
            'effective_throughput': size / elapsed / 1e6
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
                'total_time': elapsed,
                'effective_throughput': size / elapsed / 1e6
            }

    def run_benchmarks(self, iterations=3):
        self.warmup()
        results = defaultdict(list)
        
        for size in self.sizes:
            print(f"\nBenchmarking with size: {size:,}")
            for i in range(iterations):
                print(f"\nIteration {i+1}/{iterations}")

                result = self.benchmark_pytorch_cpu(size)
                results[size].append(result)
                print(f"PyTorch CPU - Size: {size:,}")
                print(f"  Total time: {result['total_time']:.5f}s")
                print(f"  Throughput: {result['effective_throughput']:.2f} M/s")

                if self.mps_available:
                    result = self.benchmark_pytorch_mps(size)
                    if result:
                        results[size].append(result)
                        print(f"PyTorch MPS - Size: {size:,}")
                        print(f"  Compute time: {result['compute_time']:.5f}s")
                        print(f"  Transfer to GPU: {result['transfer_times']['to_gpu_time']:.5f}s")
                        print(f"  Transfer to CPU: {result['transfer_times']['to_cpu_time']:.5f}s")
                        print(f"  Compute throughput: {result['compute_throughput']:.2f} M/s")
                        print(f"  Effective throughput: {result['effective_throughput']:.2f} M/s")
                
                result = self.benchmark_tensorflow(size)
                results[size].append(result)
                print(f"TensorFlow - Size: {size:,}")
                print(f"  Total time: {result['total_time']:.5f}s")
                print(f"  Throughput: {result['effective_throughput']:.2f} M/s")
                
        return results

    
    def save_results(self, results, filename):
        import json
        from datetime import datetime
        import numpy as np
        
        # Helper function to convert numpy types to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Format results for JSON
        formatted_results = {
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        for size in self.sizes:
            formatted_results['data'][str(size)] = []
            size_results = results[size]
            for r in size_results:
                result_dict = {
                    'method': r['method'],
                    'device': r['device'],
                    'size': convert_to_serializable(size)
                }
                
                # Include all timing data for MPS results
                if r['method'] == 'pytorch' and r['device'] == 'mps':
                    result_dict.update({
                        'compute_time': convert_to_serializable(r['compute_time']),
                        'transfer_to_gpu': convert_to_serializable(r['transfer_times']['to_gpu_time']),
                        'transfer_to_cpu': convert_to_serializable(r['transfer_times']['to_cpu_time']),
                        'compute_throughput': convert_to_serializable(r['compute_throughput']),
                        'effective_throughput': convert_to_serializable(r['effective_throughput'])
                    })
                else:
                    result_dict.update({
                        'total_time': convert_to_serializable(r['total_time']),
                        'effective_throughput': convert_to_serializable(r['effective_throughput'])
                    })
                
                formatted_results['data'][str(size)].append(result_dict)

        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(formatted_results, f, indent=2)
    
    def print_results(self, results):
        print("\nRNG Performance Benchmark Results:")
        print("-" * 120)
        print(f"{'Size':<14} {'Method':<12} {'Device':<8} {'Avg Time (ms)':<14} {'Throughput (M/s)':<16} {'% Transfer time':<12}")
        print("-" * 120)
        
        for size in self.sizes:
            size_results = results[size]
            by_method = defaultdict(list)
            for r in size_results:
                key = (r['method'], r['device'])
                by_method[key].append(r)
            
            for (method, device), method_results in by_method.items():
                avg_time = sum(r['total_time'] for r in method_results) / len(method_results) * 1000 #convert to ms
                avg_throughput = sum(r['effective_throughput'] for r in method_results) / len(method_results)
                if method == 'pytorch' and device =='mps':
                    avg_percent_transfer_time = sum(r['transfer_times']['total_transfer_time'] / r['total_time'] * 100 for r in method_results) / len(method_results)
                    print(f"{size:<14,} {method:<12} {device:<8} {avg_time:<14.3f} {avg_throughput:<15.2f} {avg_percent_transfer_time:<12.2f}")
                else:
                    print(f"{size:<14,} {method:<12} {device:<8} {avg_time:<14.3f} {avg_throughput:<15.2f}")
                

if __name__ == "__main__":
    benchmark = RNGBenchmarkV4()
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)
    benchmark.save_results(results, 'results/rng_benchmark.json')