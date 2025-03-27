import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BingoCNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=128,
        cnn_channels=[32, 64],
        kernel_size=3,
        cost_embed_dim=16
    ):
        super().__init__(observation_space, features_dim)

        self.board_shape = observation_space['board'].shape
        self.pattern_shape = observation_space['pattern'].shape
        input_channels = 2  # board + pattern

        layers = []
        in_channels = input_channels
        for out_channels in cnn_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.zeros((1, input_channels, *self.board_shape))
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.cost_fc = nn.Sequential(
            nn.Linear(1, cost_embed_dim),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + cost_embed_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        board = observations["board"].float()
        pattern = observations["pattern"].float()
        cost = observations["cost"].float().reshape(-1, 1)

        x = torch.stack([board, pattern], dim=1)  # (B, 2, N, N)
        cnn_out = self.cnn(x)
        cost_feat = self.cost_fc(cost)
        combined = torch.cat([cnn_out, cost_feat], dim=1)
        return self.linear(combined)