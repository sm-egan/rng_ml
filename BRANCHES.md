# Branch Management

## Active Branches

### aes
Primary branch implementing and evaluating hardware-accelerated cryptographically secure RNG.
- Focuses on AES-based random number generation with hardware acceleration
- Expanded benchmarking capabilities, including more detailed noise generation timing and saving run results
- Organized directory structure with examples folder
- Additional validation of RNG distributions

Key files:
- aes_prng.py: AES-based CSPRNG implementation
- dpsgd_rng_benchmark.py: DP-SGD implementation with AES support
- distribution_check.py: Statistical comparison of RNG outputs
- hwaccel_aes_benchmark.py: Hardware acceleration benchmarks
- examples/: Directory containing benchmark notebooks for different platforms

### standard-prng
Reference branch containing original benchmarking framework for evaluating standard PRNG performance in ML applications.
- Standard implementation using PyTorch's default PRNGs
- Single benchmark script (rng_benchmark.py) for basic RNG performance testing
- Simple directory structure with benchmarks in root directory

Key files:
- rng_benchmark.py: Core benchmarking logic
- dpsgd_rng_benchmark.py: Original DP-SGD implementation
- dpsgd_rng_benchmark.ipynb: Basic benchmark notebook

## Performance Considerations
- PRNG vs AES-CSPRNG throughput
  * It's difficult to compare apples to apples because the AES Generator has to be run on CPU. But running hwaccel_aes_benchmark.py will give you an idea of raw RNG performance on CPU for PyTorch default, AES w/o HW acceleration (PyCryptodome backend), and AES w HW acceleration (cryptography backend)
- The DPSGD benchmark runs slightly slower (60-65 ms vs. 55-60 ms) with the AES version, even with noise generation disabled. This is likely because of additional checks to ensure you're running on the correct device, as the AES Generator currently runs only on CPU.

## Branch History Note
The repository initially used a 'main' branch which was later split into more descriptively named branches 'aes' and 'standard-prng' to better reflect their purposes. The 'aes' branch serves as the primary development branch, while 'standard-prng' preserves the original PRNG implementation for performance comparisons.

## Future Integration Plans
Currently maintaining branches separately to preserve implementation differences and performance characteristics. aes will likely be merged to main, with any conflicts overwritten by aes version, once AES Generator implementation is stabilized.