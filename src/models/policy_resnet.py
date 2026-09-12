import torch
import torch.nn as nn


NUM_INPUT_PLANES = 18
NUM_CHANNELS = 64
NUM_RESIDUAL_BLOCKS = 3
NUM_MOVES = 1880


class ResidualBlock(nn.Module):
    """
    A simple residual block:

        input
          |
        Conv
        ReLU
        Conv
          |
        + input
          |
        ReLU

    This lets the network become deeper while still preserving
    information from the original board representation.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)

        x = x + residual
        x = self.relu(x)

        return x


class HumanChessResNetPolicy(nn.Module):
    """
    HumanChess V2 residual policy network.

    Input:
        [batch, 18, 8, 8]

    Output:
        [batch, 1880]
    """

    def __init__(
        self,
        input_planes=NUM_INPUT_PLANES,
        channels=NUM_CHANNELS,
        num_blocks=NUM_RESIDUAL_BLOCKS,
        num_moves=NUM_MOVES,
    ):
        super().__init__()

        # Initial board feature extraction.
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                input_planes,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # Residual tower.
        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Policy head.
        #
        # We keep this deliberately simple for the first V2 experiment.
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                channels,
                32,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                32 * 8 * 8,
                num_moves,
            ),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_tower(x)
        x = self.policy_head(x)

        return x


if __name__ == "__main__":
    model = HumanChessResNetPolicy()

    test_input = torch.zeros(
        2,
        18,
        8,
        8,
    )

    output = model(test_input)

    print(model)
    print()
    print("Input shape: ", test_input.shape)
    print("Output shape:", output.shape)