#!/bin/bash

# Show usage if no script provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

# Get the input script name
input_script="$1"

# Check if the script exists
if [ ! -f "$input_script" ]; then
    echo "Error: Script '$input_script' not found"
    exit 1
fi

# Extract base name without .py extension
# ${input_script%.py} removes .py from the end
# ${var##*/} removes everything up to the last / (the path)
base_name=$(basename "${input_script%.py}")

# Create results directory if it doesn't exist
mkdir -p results

# Generate timestamp in format YYYYMMDD_HHMMSS
timestamp=$(date "+%Y%m%d_%H%M%S")

# Run the benchmark and save output
python "$input_script" > "results/${base_name}_${timestamp}.txt" 2>&1

# Print location of results file
echo "Results saved to: results/${base_name}_${timestamp}.txt"