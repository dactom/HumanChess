import torch
import torch.nn as nn


class HumanChessPolicy(nn.Module):
    def __init__(self, num_moves=1880):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=18,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                512,
            ),
            nn.ReLU(),

            nn.Linear(
                512,
                num_moves,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.policy_head(x)
        return x
