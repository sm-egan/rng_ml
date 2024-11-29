import torch
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from typing import Optional, Tuple
from collections import deque

class BatchedAESRandomGenerator:
    """AES-CTR based RNG with batch generation and caching"""
    def __init__(
        self, 
        device: Optional[torch.device] = None,
        batch_size: int = 10_000_000,  # Generate 10M numbers at a time
        max_cache_batches: int = 2
    ):
        self.device = device if device is not None else torch.device('cpu')
        self.batch_size = batch_size
        self.max_cache_batches = max_cache_batches
        
        # Initialize AES
        self.key = get_random_bytes(16)
        self.nonce = get_random_bytes(8)
        self._init_cipher()
        
        # Initialize cache
        self.cache = deque(maxlen=max_cache_batches)
        self._current_batch = None
        self._current_idx = 0
        
    def _init_cipher(self):
        self.cipher = AES.new(self.key, AES.MODE_CTR, nonce=self.nonce)
    
    def _generate_batch(self) -> torch.Tensor:
        """Generate a batch of random numbers"""
        # Generate random bytes for the whole batch at once
        random_bytes = self.cipher.encrypt(b'\0' * (self.batch_size * 4))
        
        # Convert to float tensor efficiently
        array = np.frombuffer(random_bytes, dtype=np.uint32).astype(float)
        array /= np.uint32(0xffffffff)
        
        # Convert to tensor
        return torch.from_numpy(array)
    
    def _ensure_batch_available(self, size: int):
        """Ensure we have enough random numbers available"""
        if self._current_batch is None or self._current_idx + size > self.batch_size:
            # Generate new batch if cache is empty
            if not self.cache:
                self.cache.append(self._generate_batch())
            self._current_batch = self.cache.popleft()
            self._current_idx = 0
            
            # Pre-generate next batch if cache isn't full
            if len(self.cache) < self.max_cache_batches:
                self.cache.append(self._generate_batch())
    
    def rand(self, *size: int) -> torch.Tensor:
        """Generate uniform random float tensor with values in [0, 1)"""
        n_values = np.prod(size)
        self._ensure_batch_available(n_values)
        
        # Extract values from current batch
        values = self._current_batch[self._current_idx:self._current_idx + n_values]
        self._current_idx += n_values
        
        return values.reshape(size)
    
    def randn(self, *size: int) -> torch.Tensor:
        """Generate random tensor from standard normal distribution"""
        n_values = np.prod(size)
        self._ensure_batch_available(2 * n_values)  # Need twice as many for Box-Muller
        
        # Extract values for Box-Muller transform
        u1 = self._current_batch[self._current_idx:self._current_idx + n_values]
        self._current_idx += n_values
        u2 = self._current_batch[self._current_idx:self._current_idx + n_values]
        self._current_idx += n_values
        
        # Box-Muller transform
        angle = 2 * np.pi * u1
        radius = torch.sqrt(-2 * torch.log(u2))
        return (radius * torch.cos(angle)).reshape(size)

# Derived class for MPS support
class BatchedMPSAESRandomGenerator(BatchedAESRandomGenerator):
    """MPS-compatible version that ensures float32 output"""
    def rand(self, *size: int) -> torch.Tensor:
        return super().rand(*size).to(dtype=torch.float32).to(self.device)
    
    def randn(self, *size: int) -> torch.Tensor:
        return super().randn(*size).to(dtype=torch.float32).to(self.device)
