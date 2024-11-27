import torch
import torch.nn as nn 
from opacus.layers import DPMultiheadAttention
from dataclasses import dataclass

@dataclass
class ModelConfig:
    batch_size: int = 32
    # For transformer
    sequence_length: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    # For CNN
    image_size: int = 32
    in_channels: int = 3
    model_type: str = "transformer"  # or "resnet"

class DPTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer using DP-compatible multihead attention"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048):
        super().__init__()
        
        # Replace nn.MultiheadAttention with DPMultiheadAttention
        self.self_attn = DPMultiheadAttention(
            d_model,
            nhead,
            dropout=0.0,  # Disable dropout for benchmarking
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention block
        attn_out = self.self_attn(x, x, x)[0]
        x = self.norm1(x + attn_out)
        
        # Feed-forward block
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        
        return x

class DPTransformerEncoder(nn.Module):
    """DP-compatible transformer encoder model for benchmarking"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Create stack of DP-compatible transformer encoder layers
        self.layers = nn.ModuleList([
            DPTransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.classifier = nn.Linear(config.hidden_dim, 2)  # Binary classification
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)

class BasicBlock(nn.Module):
    """Basic ResNet block with DP compatibility"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, planes)  # GroupNorm instead of BatchNorm for DP
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, self.expansion*planes)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class DPResNet(nn.Module):
    """DP-compatible ResNet architecture for benchmarking"""
    def __init__(self, block, num_blocks, num_classes=2):
        super(DPResNet, self).__init__()
        self.in_planes = 64

        # Adjusted for smaller input size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, 64)  # GroupNorm for DP compatibility
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.activation = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Function to create ResNet18
def create_dp_resnet18():
    return DPResNet(BasicBlock, [2, 2, 2, 2])
