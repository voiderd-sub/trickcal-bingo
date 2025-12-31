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
    ResNet-style CNN feature extractor for Bingo environment.
    
    Input channels:
        - board (7x7)
        - current pattern (7x7)
        - stored pattern (7x7)
    
    Additional scalar features:
        - has_stored (0 or 1)
    """
    def __init__(
        self,
        observation_space,
        features_dim=256,
        hidden_channels=64,
        num_res_blocks=3,
        kernel_size=3,
        scalar_embed_dim=32
    ):
        super().__init__()
        self._features_dim = features_dim

        self.board_shape = observation_space['board'].shape
        input_channels = 3  # board + pattern + stored_pattern

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU()
        )
        nn.init.orthogonal_(self.initial_conv[0].weight, gain=np.sqrt(2))

        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])

        # Global average pooling + flatten
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Scalar feature embedding
        # has_stored: 0 or 1
        self.scalar_fc = nn.Sequential(
            nn.Linear(1, scalar_embed_dim),
            nn.GELU(),
            nn.Linear(scalar_embed_dim, scalar_embed_dim),
            nn.GELU()
        )
        
        # Initialize scalar FC
        for layer in self.scalar_fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        # Final linear layer
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_channels + scalar_embed_dim, features_dim),
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
        # Extract tensors
        board = observations["board"].float()
        pattern = observations["pattern"].float()
        stored_pattern = observations["stored_pattern"].float()
        # Scalar features are already shape (B, 1) from Box space
        has_stored = observations["has_stored"].float()

        # Stack as 3 channels: (B, 3, H, W)
        x = torch.stack([board, pattern, stored_pattern], dim=1)

        # CNN forward
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)  # (B, hidden_channels)

        # Scalar features
        scalar_embed = self.scalar_fc(has_stored)

        # Combine and output
        combined = torch.cat([x, scalar_embed], dim=1)
        return self.final_fc(combined)


if __name__ == "__main__":
    from gymnasium import spaces
    
    # Test observation space
    board_size = 7
    obs_space = spaces.Dict({
        "board": spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int8),
        "pattern": spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int8),
        "stored_pattern": spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int8),
        "has_stored": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
    })
    
    extractor = BingoCNNExtractor(obs_space, features_dim=256)
    print(f"Model created: {sum(p.numel() for p in extractor.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    dummy_obs = {
        "board": torch.zeros(batch_size, board_size, board_size),
        "pattern": torch.zeros(batch_size, board_size, board_size),
        "stored_pattern": torch.zeros(batch_size, board_size, board_size),
        "has_stored": torch.zeros(batch_size, 1),
    }
    
    output = extractor(dummy_obs)
    print(f"Output shape: {output.shape}")  # Should be (4, 256)