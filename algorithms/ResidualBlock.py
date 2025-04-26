import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        return out
