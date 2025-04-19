import torch

from torch import nn

from .base_neural_function import BaseNeuralFunction


class QFunction(BaseNeuralFunction):
    def __init__(self, state_dim: int, action_n: int, inner_layer=256):
        super().__init__(state_dim, action_n, inner_layer)

    def get_target_copy(self, device: torch.device) -> "QFunction":
        target_q_function: nn.Module = QFunction(
            self.state_dim, self.action_n, self.inner_layer
        )
        target_q_function = target_q_function.to(device)

        state_dict = self.state_dict()
        target_q_function.load_state_dict(state_dict)

        for param in target_q_function.parameters():
            param.requires_grad = False

        return target_q_function

    def update(self, source_q_function: "QFunction", tau: float = 0.1) -> None:
        target_dict = self.state_dict()
        for name, param in source_q_function.named_parameters():
            target_dict[name] = tau * param.data + (1 - tau) * target_dict[name]
        self.load_state_dict(target_dict)
