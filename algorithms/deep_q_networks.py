import torch
from torch import nn
import numpy as np

import random

from .q_function import QFunction


class DQN(nn.Module):
    def __init__(
        self,
        q_function: QFunction,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        epsilon_decrease=0.02,
        epsilon_min=0.01,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_function = q_function.to(device=self.device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    @torch.no_grad()
    def get_action(self, state, info):
        mask = info["actions"]
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.q_function.predict(state).squeeze()
        masked_q_values = q_values[mask]
        masked_argmax_action = torch.argmax(masked_q_values)
        probs = self.epsilon * np.ones_like(mask) / len(mask)
        probs[masked_argmax_action] += 1 - self.epsilon
        masked_action = np.random.choice(mask, p=probs)
        return masked_action

    @torch.no_grad()
    def get_max_q_values(self, next_states, next_guides):
        return torch.max(self.q_function(next_states).take_along_dim(next_guides, dim=1),
                         dim=1).values

    def get_padded_to_action_n(self, guide: torch.Tensor):
        if len(guide) == 0:
            return torch.zeros(self.q_function.action_n, dtype=torch.long)
        return torch.nn.functional.pad(guide,
                                       pad=(0, self.q_function.action_n - guide.numel()),
                                       mode="constant",
                                       value=guide[0].item())

    def fit(self, state, info, action, reward, done, next_state, next_info):
        if not self.training:
            return
        next_guide = torch.tensor(next_info['actions'])
        self.memory.append(
            [
                torch.tensor(state),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward),
                torch.tensor(int(done)),
                torch.tensor(next_state),
                self.get_padded_to_action_n(next_guide)
            ]
        )

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states, next_guides = map(
            torch.stack, list(zip(*batch))
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        next_guides = next_guides.to(self.device)

        max_q_values = self.get_max_q_values(next_states, next_guides)
        targets = rewards + self.gamma * (1 - dones) * max_q_values
        q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

        loss = torch.mean((q_values - targets.detach()) ** 2)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.epsilon -= self.epsilon_decrease
        self.epsilon = max(self.epsilon_min, self.epsilon)

        return loss.cpu().detach().numpy()

    def train(self, mode: bool = True, epsilon: float = 0):
        super().train(mode)
        self.epsilon = epsilon if mode else 0
        return self

    def eval(self):
        return self.train(False)
    
    @torch.no_grad()
    def get_value(self, state, info):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        guide = torch.tensor(info["actions"], dtype=torch.long)
        q_values = self.q_function.forward(state).squeeze()
        return q_values[guide].max().cpu().item()


class TargetDQN(DQN):
    def __init__(
        self,
        q_function: QFunction,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        target_update=10,
        epsilon_decrease=0.01,
        epsilon_min=0.01,
    ):
        super().__init__(
            q_function, gamma, lr, batch_size, epsilon_decrease, epsilon_min
        )
        self.target_q_function = self.q_function.get_target_copy(self.device)
        self.target_update = target_update
        self.fit_calls = 0

    def update_target(self):
        pass

    @torch.no_grad()
    def get_max_q_values(self, next_states, next_guides):
        return torch.max(self.target_q_function(next_states).take_along_dim(next_guides, dim=1),
                         dim=1).values

    def fit(self, state, info, action, reward, done, next_state, next_actions):
        if not self.training:
            return
        loss = super().fit(state, info, action, reward, done, next_state, next_actions)

        self.fit_calls += 1
        if self.fit_calls >= self.target_update:
            self.update_target()
            self.fit_calls = 0

        return loss


class HardTargetDQN(TargetDQN):
    def update_target(self):
        state_dict = self.q_function.network.state_dict()
        self.target_q_function.network.load_state_dict(state_dict)


class SoftTargetDQN(TargetDQN):
    def __init__(
        self,
        q_function: QFunction,
        gamma=0.99,
        lr=1e-3,
        tau=0.1,
        batch_size=64,
        epsilon_decrease=0.01,
        epsilon_min=0.01,
    ):
        super().__init__(
            q_function,
            gamma,
            lr,
            batch_size,
            1,
            epsilon_decrease,
            epsilon_min,
        )
        self.tau = tau

    def update_target(self):
        # theta' = tau * theta + (1 - tau) * theta'
        target_dict = self.target_q_function.state_dict()
        for name, param in self.q_function.named_parameters():
            target_dict[name] = (
                self.tau * param.data + (1 - self.tau) * target_dict[name]
            )
        self.target_q_function.load_state_dict(target_dict)


class DoubleDQN(SoftTargetDQN):
    def get_max_q_values(self, next_states, next_guides):
        next_states_q = self.q_function.forward(next_states).take_along_dim(next_guides, dim=1)
        best_actions = torch.argmax(next_states_q, axis=1)
        best_actions = next_guides[torch.arange(self.batch_size), best_actions]

        max_q_values = self.target_q_function(next_states)[
            np.arange(0, self.batch_size), best_actions
        ]

        return max_q_values
