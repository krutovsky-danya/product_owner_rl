import torch
from torch import nn
import numpy as np

import random


class QFunction(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_n),
        )

    def forward(self, states):
        return self.network(states)


class DQN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        epsilon_decrease=0.02,
        epsilon_min=0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = QFunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    @torch.no_grad()
    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    @torch.no_grad()
    def get_max_q_values(self, next_states):
        return torch.max(self.q_function(next_states), dim=1).values

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = map(
            torch.tensor, list(zip(*batch))
        )

        max_q_values = self.get_max_q_values(next_states)
        targets = rewards + self.gamma * (1 - dones) * max_q_values
        q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

        loss = torch.mean((q_values - targets.detach()) ** 2)
        loss.backward()
        self.optimzaer.step()
        self.optimzaer.zero_grad()

        self.epsilon -= self.epsilon_decrease
        self.epsilon = max(self.epsilon_min, self.epsilon)


class TargetDQN(DQN):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        target_update=10,
        epsilon_decrease=0.01,
        epilon_min=0.01,
    ):
        super().__init__(
            state_dim, action_dim, gamma, lr, batch_size, epsilon_decrease, epilon_min
        )
        self.target_q_function = QFunction(state_dim, action_dim)

        state_dict = self.q_function.network.state_dict()
        self.target_q_function.network.load_state_dict(state_dict)

        for p in self.target_q_function.network.parameters():
            p.requires_grad = False

        self.target_update = target_update
        self.fit_calls = 0
    
    def update_target(self):
        pass

    @torch.no_grad()
    def get_max_q_values(self, next_states):
        return torch.max(self.target_q_function(next_states), dim=1).values

    def fit(self, state, action, reward, done, next_state):
        super().fit(state, action, reward, done, next_state)

        self.fit_calls += 1
        if self.fit_calls >= self.target_update:
            self.update_target()
            self.fit_calls = 0


class HardTargetDQN(TargetDQN):
    def update_target(self):
        state_dict = self.q_function.network.state_dict()
        self.target_q_function.network.load_state_dict(state_dict)


class SoftTargetDQN(TargetDQN):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        tau=0.1,
        batch_size=64,
        epsilon_decrease=0.01,
        epilon_min=0.01,
    ):
        super().__init__(
            state_dim,
            action_dim,
            gamma,
            lr,
            batch_size,
            1,
            epsilon_decrease,
            epilon_min,
        )
        self.tau = tau

    def update_target(self):
        # theta' = tau * theta + (1 - tau) * theta'
        target_dict = self.target_q_function.state_dict()
        for name, param in self.q_function.named_parameters():
            target_dict[name] = self.tau * param.data + (1 - self.tau) * target_dict[name]
        self.target_q_function.load_state_dict(target_dict)


class DoubleDQN(SoftTargetDQN):
    def get_max_q_values(self, next_states):
        next_states_q = self.q_function(next_states)
        best_actions = torch.argmax(next_states_q, axis=1)

        max_q_values = self.target_q_function(next_states)[
            np.arange(0, self.batch_size), best_actions
        ]

        return max_q_values
