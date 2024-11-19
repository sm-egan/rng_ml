#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Generate timestamp in format YYYYMMDD_HHMMSS
timestamp=$(date "+%Y%m%d_%H%M%S")

# Run the benchmark and save output silently
python rng_benchmark.py > "results/rng_benchmark_${timestamp}.txt" 2>&1

# Print only the location of results file
echo "Results saved to: results/rng_benchmark_${timestamp}.txt"
