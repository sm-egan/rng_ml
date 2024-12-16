import torch
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from typing import Optional, Tuple, Union, Dict, Any
from collections import deque
from opacus import PrivacyEngine
import warnings
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

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
        
        return values.reshape(size).to(self.device)
    
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
        return (radius * torch.cos(angle)).reshape(size).to(self.device)

    def get_state(self) -> Dict[str, Any]:
        """Gets the generator state"""
        return {
            'key': self.key,
            'nonce': self.nonce,
            'current_batch': self._current_batch,
            'current_idx': self._current_idx,
            'cache': list(self.cache)
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Sets the generator state"""
        self.key = state['key']
        self.nonce = state['nonce']
        self._init_cipher()
        self._current_batch = state['current_batch']
        self._current_idx = state['current_idx']
        self.cache = deque(state['cache'], maxlen=self.max_cache_batches)

# Derived class for MPS support
class BatchedMPSAESRandomGenerator(BatchedAESRandomGenerator):
    """MPS-compatible version that ensures float32 output"""
    def rand(self, *size: int) -> torch.Tensor:
        return super().rand(*size).to(dtype=torch.float32).to(self.device)
    
    def randn(self, *size: int) -> torch.Tensor:
        return super().randn(*size).to(dtype=torch.float32).to(self.device)

class AESGenerator(torch.Generator):
    """PyTorch Generator implementation using AES-CTR for random number generation"""
    def __init__(self):
        # Call parent init without arguments
        super().__init__()
        
        # Initialize our batched generator
        self._generator = BatchedAESRandomGenerator(device='cpu')
        self._seed = None
    
    def manual_seed(self, seed: int) -> 'AESGenerator':
        self._seed = seed
        return self
        
    def initial_seed(self) -> int:
        if self._seed is None:
            self._seed = int.from_bytes(get_random_bytes(8), byteorder='big')
        return self._seed
    
    def seed(self) -> int:
        self._seed = int.from_bytes(get_random_bytes(8), byteorder='big')
        return self._seed

    def get_state(self) -> Dict[str, Any]:
        return {
            'seed': self._seed,
            'generator_state': self._generator.get_state()
        }
    
    def set_state(self, state: Dict[str, Any]) -> 'AESGenerator':
        self._seed = state['seed']
        self._generator.set_state(state['generator_state'])
        return self

    def random_(self, tensor):
        """Fill tensor with random values from uniform distribution [0, 1)"""
        with torch.no_grad():
            # Generate random values on CPU
            random_tensor = self._generator.rand(*tensor.size())
            
            # Move to target device and copy to input tensor
            tensor.copy_(random_tensor.to(tensor.device))
            
        return tensor

class HWAccelBatchedAESRandomGenerator(BatchedAESRandomGenerator):
    """AES-CTR RNG that uses hardware acceleration when available"""
    def __init__(self, device: Optional[torch.device] = None,
                 batch_size: int = 10_000_000,
                 max_cache_batches: int = 2):
        #super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.batch_size = batch_size
        self.max_cache_batches = max_cache_batches
        
        # Initialize AES with appropriate nonce size
        self.key = get_random_bytes(16)
        # CTR mode requires 16-byte nonce for cryptography library
        self.nonce = get_random_bytes(16)  # Changed from 8 to 16
        self._init_cipher()
        
        # Initialize cache
        self.cache = deque(maxlen=max_cache_batches)
        self._current_batch = None
        self._current_idx = 0

    def _init_cipher(self):
        """Override cipher initialization to use cryptography library"""
        # Create cipher using hardware backend
        self.cipher = Cipher(
            algorithms.AES(self.key),
            modes.CTR(self.nonce),  # No need to pad, using 16 bytes directly
            backend=default_backend()
        ).encryptor()
    
    def _generate_batch(self) -> torch.Tensor:
        """Generate a batch of random numbers using hardware acceleration"""
        # Generate random bytes using hardware-accelerated AES
        random_bytes = self.cipher.update(b'\0' * (self.batch_size * 4))
        
        # Convert to float tensor efficiently (same as parent)
        array = np.frombuffer(random_bytes, dtype=np.uint32).astype(float)
        array /= np.uint32(0xffffffff)
        
        return torch.from_numpy(array)

# Create a hardware-accelerated version of the main Generator
class HWAccelAESGenerator(AESGenerator):
    """AES Generator that uses hardware acceleration"""
    def __init__(self):
        #super().__init__()
        # Override the generator to use hardware-accelerated version
        self._generator = HWAccelBatchedAESRandomGenerator(device='cpu')

class AESPrivacyEngine(PrivacyEngine):
    """Privacy Engine that uses AES-CTR for secure random number generation"""
    def __init__(
        self, 
        *,
        accountant: str = "prv",
        secure_mode: bool = True,
        device: Optional[Union[str, torch.device]] = None
    ):
        # Initialize accountant but skip parent's secure RNG setup
        self.accountant = self._set_accountant(accountant)
        self.secure_mode = secure_mode
        self.secure_rng = None
        self.dataset = None
        
        if self.secure_mode:
            # Create CPU generator - it will handle device transfers internally
            print("Secure mode activated...")
            self.secure_rng = HWAccelAESGenerator()
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_mode`` turned on."
            )

    def _set_accountant(self, accountant: str):
        """Helper method to set up the accountant"""
        from opacus.accountants import create_accountant
        return create_accountant(mechanism=accountant)
