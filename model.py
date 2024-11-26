# model.py
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    batch_size: int = 32
    sequence_length: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2

class TransformerEncoder(nn.Module):
    """Simple transformer encoder model for benchmarking"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Output projection
        self.classifier = nn.Linear(config.hidden_dim, 2)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)
