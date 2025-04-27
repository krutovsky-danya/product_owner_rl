from algorithms.base_neural_function import BaseNeuralFunction


import torch
from torch import nn


class PolicyFunction(BaseNeuralFunction):
    def __init__(self, state_dim, action_n, inner_layer=256):
        super().__init__(state_dim, action_n, inner_layer)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, states):
        logits = super().forward(states)
        return self.softmax(logits)

    @torch.no_grad()
    def predict(self, state):
        logits = super().predict(state)
        return self.softmax(logits)

    def forward_guided(self, states, guides) -> torch.Tensor:
        logits: torch.Tensor = super().forward(states)
        # masked_fill is out-of-place version of masked_fill_
        # (in code it usually looks like this: tensor[mask] = value
        # where mask is a Tensor of booleans)
        logits = logits.masked_fill(~guides, -torch.inf)
        return self.softmax(logits)

    def get_normalized_guided_logits(self, states, guides) -> torch.Tensor:
        """Returns normalized logits for guided actions."""
        logits: torch.Tensor = super().forward(states)
        logits = logits.masked_fill(~guides, -torch.inf)
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return logits

    @torch.no_grad()
    def predict_guided(self, state, guide) -> torch.Tensor:
        logits = super().predict(state)
        logits[~guide] = -torch.inf
        return self.softmax(logits)
