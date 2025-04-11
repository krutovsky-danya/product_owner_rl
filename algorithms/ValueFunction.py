import torch

from algorithms.base_neural_function import BaseNeuralFunction


class ValueFunction(BaseNeuralFunction):
    def __init__(self, state_dim, inner_layer=256):
        super().__init__(state_dim, 1, inner_layer)

    def forward(self, states):
        return super().forward(states).squeeze(dim=1)

    @torch.no_grad()
    def predict(self, state):
        return super().predict(state).squeeze(dim=1)
