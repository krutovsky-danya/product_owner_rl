import torch
from torch import nn

from .ResidualBlock import ResidualBlock


class BaseNeuralFunction(nn.Module):
    def __init__(self, state_dim: int, action_n: int, inner_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.inner_dim = inner_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, inner_dim),
            nn.ReLU(),
            ResidualBlock(inner_dim),
            nn.Linear(inner_dim, action_n),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for a given state tensor.

        Args:
            state (torch.Tensor): A tensor representing the state, with shape (state_dim,).

        Returns:
            torch.Tensor: A tensor representing the predicted output, with shape (1, action_n).
        """
        result: torch.Tensor = self(state.unsqueeze(0))
        return result.squeeze(0)
