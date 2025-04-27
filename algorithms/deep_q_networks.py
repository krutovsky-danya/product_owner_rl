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

    def get_probs(self, q_values: torch.Tensor, guides: torch.Tensor) -> torch.Tensor:
        masked_q_values = q_values.masked_fill(~guides, -np.inf)
        argmax_action = torch.argmax(masked_q_values)
        probs = torch.tensor(guides, dtype=torch.float, device=self.device)
        probs = self.epsilon * probs / guides.sum()
        probs[argmax_action] += 1 - self.epsilon
        return probs

    @torch.no_grad()
    def get_action(self, state, info):
        guides = torch.tensor(info["actions"], dtype=torch.bool, device=self.device)
        state = torch.tensor(state, dtype=torch.float, device=self.device)

        probs = self.get_probs(self.q_function.predict(state), guides)
        masked_action = np.random.choice(guides.size(dim=0), p=probs.cpu().numpy())
        return masked_action

    @torch.no_grad()
    def get_max_q_values(self, states: torch.Tensor, guides: torch.Tensor):
        q_values: torch.Tensor = self.q_function(states)
        q_values = q_values.masked_fill(~guides, -np.inf)
        return torch.max(q_values, dim=1).values

    def fit(self, state, info, action, reward, done, next_state, next_info):
        if not self.training:
            return
        self.memory.append(
            [
                torch.tensor(state),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward),
                torch.tensor(done, dtype=torch.long),
                torch.tensor(next_state),
                torch.tensor(next_info["actions"], dtype=torch.bool),
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
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        guides = torch.tensor(
            info["actions"], dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        return self.get_max_q_values(state, guides).squeeze()


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
    def get_max_q_values(self, states: torch.Tensor, guides: torch.Tensor):
        q_values: torch.Tensor = self.target_q_function(states)
        q_values = q_values.masked_fill(~guides, -np.inf)
        return torch.max(q_values, dim=1).values

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
        self.target_q_function.update(self.q_function, self.tau)


class DoubleDQN(SoftTargetDQN):
    def get_max_q_values(self, states: torch.Tensor, guides: torch.Tensor):
        q_values = self.q_function.forward(states).masked_fill(~guides, -np.inf)
        best_actions: torch.Tensor = torch.argmax(q_values, axis=1)

        target_q_values = self.target_q_function.forward(states)
        max_q_values = target_q_values.gather(1, best_actions.unsqueeze(1))

        return max_q_values.squeeze(1)
