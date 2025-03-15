import torch
from torch import nn


class BaseNeuralFunction(nn.Module):
    def __init__(self, state_dim: int, action_n: int, inner_layer=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.inner_layer = inner_layer

        self.network = nn.Sequential(
            nn.Linear(state_dim, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, action_n),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        self.network.eval()
        result = self.network(state.unsqueeze(0))
        self.network.train(True)
        return result
