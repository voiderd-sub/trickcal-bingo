"""
Policy network for Bingo RL agent.

Uses CNN with 3 channels: board + pattern_0 + pattern_1
All three 7x7 grids go through the same CNN for spatial reasoning.

Architecture:
- Input: 3 channels (board, pattern_0, pattern_1) as 7x7 grids
- CNN with residual blocks
- MLP heads for policy and value
"""

import torch
from torch import nn
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual Block with GELU and LayerNorm.
    
    Structure: x -> Conv -> LN -> GELU -> Conv -> LN -> + x -> GELU
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.ln1 = nn.GroupNorm(1, channels)  # LayerNorm equivalent for conv
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.ln2 = nn.GroupNorm(1, channels)
        self.gelu = nn.GELU()
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, gain=1.0)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.ln2(out)
        
        out = out + identity
        out = self.gelu(out)
        
        return out


class BingoCNNExtractor(nn.Module):
    """
    CNN feature extractor for Bingo environment.
    
    Input:
        - board: (B, 7, 7) binary board state
        - pattern_0: (B, 7, 7) first pattern (padded to 7x7)
        - pattern_1: (B, 7, 7) second pattern (zeros if num_patterns=1)
    
    Stacks all three as channels and processes through CNN.
    This allows the network to learn spatial relationships between
    the board state and pattern shapes.
    """
    def __init__(
        self,
        observation_space,
        features_dim=256,
        hidden_channels=64,
        num_res_blocks=3,
        kernel_size=3,
        num_patterns=1,  # For compatibility, not used in CNN approach
    ):
        super().__init__()
        self._features_dim = features_dim
        self.num_patterns = num_patterns

        # CNN: 3 channels (board + pattern_0 + pattern_1)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU()
        )
        nn.init.orthogonal_(self.initial_conv[0].weight, gain=np.sqrt(2))

        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final MLP
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_channels, features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim),
            nn.GELU()
        )
        
        for layer in self.final_fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    @property
    def features_dim(self):
        """Output feature dimension (for compatibility)."""
        return self._features_dim

    def forward(self, observations):
        # Get observations
        board = observations["board"].float()          # (B, 7, 7)
        pattern_0 = observations["pattern_0"].float()  # (B, 7, 7)
        pattern_1 = observations["pattern_1"].float()  # (B, 7, 7)
        
        # Stack as 3 channels: (B, 3, 7, 7)
        x = torch.stack([board, pattern_0, pattern_1], dim=1)
        
        # CNN
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)  # (B, hidden_channels)
        
        return self.final_fc(x)


if __name__ == "__main__":
    from gymnasium import spaces
    
    print("=== BingoCNNExtractor Test (CNN 3-channel) ===")
    
    # Test with different num_patterns
    for num_patterns in [1, 2]:
        print(f"\n--- num_patterns={num_patterns} ---")
        
        # Observation space
        obs_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern_0": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern_1": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
        })
        
        extractor = BingoCNNExtractor(
            obs_space, 
            features_dim=256,
            num_patterns=num_patterns,
        )
        
        total_params = sum(p.numel() for p in extractor.parameters())
        print(f"  Parameters: {total_params:,}")
        
        # Test forward pass
        batch_size = 4
        dummy_obs = {
            "board": torch.zeros(batch_size, 7, 7),
            "pattern_0": torch.zeros(batch_size, 7, 7),
            "pattern_1": torch.zeros(batch_size, 7, 7),
        }
        
        output = extractor(dummy_obs)
        print(f"  Output shape: {output.shape}")  # Should be (4, 256)
    
    print("\nAll tests passed!")