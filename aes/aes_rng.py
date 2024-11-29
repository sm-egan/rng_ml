import torch
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np
from typing import Optional, Union, Tuple

class AESRandomGenerator:
    """
    AES-CTR based random number generator compatible with PyTorch.
    Uses PyCryptodome's AES implementation in counter mode to generate
    a stream of random numbers.
    """
    def __init__(self, seed: Optional[bytes] = None, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device('cpu')
        # Generate key and nonce if not provided
        self.key = seed if seed is not None else get_random_bytes(16)  # AES-128
        self.nonce = get_random_bytes(8)  # 64-bit nonce
        self._init_cipher()
        
    def _init_cipher(self):
        """Initialize or reinitialize AES cipher"""
        self.cipher = AES.new(self.key, AES.MODE_CTR, nonce=self.nonce)
        
    def _generate_random_bytes(self, n_bytes: int) -> bytes:
        """Generate random bytes using AES-CTR"""
        return self.cipher.encrypt(b'\0' * n_bytes)
    
    def _bytes_to_float_array(self, random_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Convert random bytes to numpy array of float32 in [0, 1)"""
        uint32_array = np.frombuffer(random_bytes, dtype=np.uint32)
        return (uint32_array.astype(np.float32) / np.uint32(0xffffffff)).reshape(shape)
    
    def rand(self, *size: int) -> torch.Tensor:
        """Generate uniform random float tensor with values in [0, 1)"""
        n_values = np.prod(size)
        random_bytes = self._generate_random_bytes(n_values * 4)
        random_array = self._bytes_to_float_array(random_bytes, size)
        return torch.from_numpy(random_array).to(self.device)
    
    def randn(self, *size: int) -> torch.Tensor:
        """Generate random tensor from standard normal distribution using Box-Muller"""
        u1 = self.rand(*size)
        u2 = self.rand(*size)
        
        angle = 2 * np.pi * u1
        radius = torch.sqrt(-2 * torch.log(u2))
        return radius * torch.cos(angle)

    def manual_seed(self, seed: int):
        """Reset the RNG with a new seed"""
        # Convert int to bytes in a deterministic way
        seed_bytes = seed.to_bytes(16, byteorder='big')
        self.key = seed_bytes
        self._init_cipher()

# Example of how to integrate with Opacus
def patch_opacus_rng():
    """
    Patch Opacus to use our AES-based RNG.
    This needs to hook into Opacus's RNG initialization and noise generation.
    """
    from opacus.optimizers import optimizer
    
    # Store original function to restore later if needed
    original_generate_noise = optimizer._generate_noise
    
    # Create our secure generator
    secure_generator = AESRandomGenerator()
    
    def secure_generate_noise(std: float, size: torch.Size, device: torch.device) -> torch.Tensor:
        """Generate noise using our secure RNG"""
        return std * secure_generator.randn(*size).to(device)
    
    # Replace Opacus's noise generation function
    optimizer._generate_noise = secure_generate_noise
    
    return original_generate_noise  # In case we need to restore it

