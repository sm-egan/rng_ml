# Branch Management

## Active Branches

### main
Primary branch containing original benchmarking framework for evaluating RNG performance in ML applications.
- Original implementation focusing on standard PyTorch PRNGs
- Single benchmark script (rng_benchmark.py) for basic RNG performance testing
- Simple directory structure with benchmarks in root directory

Key files:
- rng_benchmark.py: Core benchmarking logic
- dpsgd_rng_benchmark.py: Original DP-SGD implementation
- dpsgd_rng_benchmark.ipynb: Basic benchmark notebook

### aes
Development branch implementing and evaluating hardware-accelerated cryptographically secure RNG.
- Focuses on AES-based random number generation with hardware acceleration
- Expanded benchmarking capabilities, including more detailed noise generation timing and saving run results
- Reorganized directory structure with examples folder
- Additional validation of RNG distributions

Key changes:
- Added new files:
  * aes_prng.py: AES-based CSPRNG implementation
  * distribution_check.py: Statistical comparison of RNG outputs. Essentially checking we can produce uniform, normal random numbers of reasonable quality from the AES generator
  * hwaccel_aes_benchmark.py: Hardware acceleration benchmarks
  * examples/: Directory containing benchmark notebooks for different platforms

- Modified files:
  * dpsgd_rng_benchmark.py: Updated to support AES-based RNG
  * run_rng_benchmark.sh: Modified for new benchmark structure
  * .gitignore: Updated for new project structure

- Removed files:
  * rng_benchmark.py: Functionality split into more specialized benchmark scripts

## Performance Considerations
- PRNG vs AES-CSPRNG throughput
  * It's difficult to compare apples to apples because the AES Generator has to be run on CPU. But running hwaccel_aes_benchmark.py will give you an idea of raw RNG performance on CPU for PyTorch default, AES w/o HW acceleration (PyCryptodome backend), and AES w HW acceleration (cryptography backend)
- The DPSGD benchmark runs slightly slower (60-65 ms vs. 55-60 ms) in the aes branch version, even with noise generation disabled. This is likely because of additional checks to ensure you're running on the correct device, as the AES Generator currently runs only on CPU.

## Future Integration Plans
Currently maintaining branches separately to preserve implementation differences and performance characteristics. aes will likely be merged to main, with any conflicts overwritten by aes version, once AES Generator implementation is stabilized.