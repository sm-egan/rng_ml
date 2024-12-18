# RNG ML: Benchmarking Random Number Generation in DP-SGD

This repository implements benchmarks to measure the overhead of cryptographically secure random number generation in differentially private stochastic gradient descent (DP-SGD) training. The benchmarks specifically focus on quantifying the performance impact of using AES-CTR (Counter Mode) for generating the noise required in DP-SGD, compared to standard pseudo-random number generation.

## Background

Differential Privacy requires cryptographically secure random number generation to maintain its privacy guarantees. However, existing DP-ML frameworks typically use standard PRNGs for performance reasons. This benchmark helps quantify the overhead of using proper cryptographic RNG in private training workflows.

## Features

- Benchmarks DP-SGD training steps with both standard and secure noise generation
- Implements hardware-accelerated AES-CTR random number generation
- Supports both ResNet-18 and Transformer architectures
- Detailed timing breakdown of forward/backward passes and noise generation
- Memory usage tracking for CPU and GPU
- Compatible with CUDA GPUs and Apple Metal (MPS)

## AES Generator Implementation

The repository includes two AES-based random number generators:

- `BatchedAESRandomGenerator`: Base implementation using AES in Counter Mode
- `HWAccelBatchedAESRandomGenerator`: Version that utilizes hardware acceleration via:
  - Intel AES-NI instructions on x86 processors
  - Secure Enclave on Apple Silicon
  
The generators use batch generation and caching to improve performance, generating blocks of 10M random numbers at a time.

## Running the Benchmark

1. First, set up your Python environment:
```bash
python -m venv dpsgd_rng_env
source dpsgd_rng_env/bin/activate  # On Unix/macOS
pip install -r requirements.txt
```

2. Run the benchmark script:
```bash
# Basic usage with default settings (Transformer model)
python dpsgd_rng_benchmark.py

# To benchmark ResNet-18
python dpsgd_rng_benchmark.py --model_type resnet

# To disable Poisson sampling
python dpsgd_rng_benchmark.py --poisson_sampling false

# To use the AES privacy engine
python dpsgd_rng_benchmark.py --privacy_engine_type aes

# To specify a different output directory
python dpsgd_rng_benchmark.py --output_dir custom_results
```

Results will be saved to the `results` directory, including:
- Detailed timing statistics
- Memory usage metrics
- Per-tensor noise generation times
- Raw data in CSV and NPZ formats

## Output Structure

The benchmark generates several files in the results directory:
```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json           # Benchmark configuration
    ├── summary.json         # Summary statistics
    ├── timing_details.csv   # Detailed timing data
    ├── noise_generation_details.csv   # Per-tensor noise timing
    ├── tensor_stats_summary.csv       # Tensor statistics
    └── all_data.npz         # All data in compressed format
```

## Citation

If you use this benchmark in your research, please cite our paper:
```bibtex
@inproceedings{egan2024,
  title={High-speed secure random number generator co-processors for privacy-preserving machine learning},
  author={Egan, Shannon},
  booktitle={Second Workshop on Machine Learning with New Compute Paradigms at NeurIPS},
  year={2024},
  url={https://openreview.net/pdf?id=8oraBCaYbm}
}
```
