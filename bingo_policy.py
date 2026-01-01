"""
Policy network for Bingo RL agent.

Uses CNN for board state and embeddings for pattern indices.
This is more efficient than encoding 5-class patterns as 7x7 images.

Architecture:
- Board: 7x7 → CNN → features (spatial understanding)
- Patterns: indices → Embedding(5, dim) → concat per pattern
- Combined features → MLP → policy/value heads
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
        - pattern_indices: (B, num_patterns) int64 pattern type indices (0-4, or -1 for empty)
    
    Uses CNN only for board (spatial reasoning needed).
    Uses embedding for patterns (only 5 types, no spatial info needed).
    """
    def __init__(
        self,
        observation_space,
        features_dim=256,
        hidden_channels=64,
        num_res_blocks=3,
        kernel_size=3,
        pattern_embed_dim=32,
        num_pattern_types=5,
        num_patterns=1,  # Can be any positive integer
    ):
        super().__init__()
        self._features_dim = features_dim
        self.num_patterns = num_patterns
        self.num_pattern_types = num_pattern_types

        # Board CNN: single channel input (just the board)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
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

        # Pattern embedding: index -> vector
        # +1 for the "empty" pattern (index -1 mapped to num_pattern_types)
        self.pattern_embedding = nn.Embedding(num_pattern_types + 1, pattern_embed_dim)
        nn.init.orthogonal_(self.pattern_embedding.weight, gain=1.0)

        # Final MLP: board features + all pattern embeddings
        combined_dim = hidden_channels + (num_patterns * pattern_embed_dim)
        self.final_fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
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
        # Board: (B, 7, 7) -> (B, 1, 7, 7)
        board = observations["board"].float().unsqueeze(1)
        
        # Pattern indices: (B, num_patterns) int64
        pattern_indices = observations["pattern_indices"].long()
        
        # Map -1 (empty) to num_pattern_types for embedding lookup
        pattern_indices = torch.where(
            pattern_indices < 0,
            torch.full_like(pattern_indices, self.num_pattern_types),
            pattern_indices
        )
        
        # CNN for board
        x = self.initial_conv(board)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        board_features = x.flatten(start_dim=1)  # (B, hidden_channels)
        
        # Embed all patterns: (B, num_patterns) -> (B, num_patterns, embed_dim)
        pattern_embeds = self.pattern_embedding(pattern_indices)
        # Flatten: (B, num_patterns * embed_dim)
        pattern_features = pattern_embeds.flatten(start_dim=1)
        
        # Combine and output
        combined = torch.cat([board_features, pattern_features], dim=1)
        return self.final_fc(combined)


if __name__ == "__main__":
    from gymnasium import spaces
    
    print("=== BingoCNNExtractor Test (with Embedding) ===")
    
    # Test with different num_patterns
    for num_patterns in [1, 2, 3, 5]:
        print(f"\n--- num_patterns={num_patterns} ---")
        
        # Observation space (simplified - pattern_indices instead of 7x7 grids)
        obs_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern_indices": spaces.Box(low=-1, high=4, shape=(num_patterns,), dtype=np.int64),
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
            "pattern_indices": torch.randint(-1, 5, (batch_size, num_patterns)),
        }
        
        output = extractor(dummy_obs)
        print(f"  Output shape: {output.shape}")  # Should be (4, 256)
    
    print("\nAll tests passed!")