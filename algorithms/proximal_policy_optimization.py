import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical


class PPO(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        batch_size=128,
        epsilon=0.2,
        epoch_n=30,
        pi_lr=1e-4,
        v_lr=5e-4,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * action_dim),
            nn.Tanh(),
        )

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

    def get_dist(self, pi_values, states_n):
        values = pi_values.reshape(states_n, self.action_dim, 2)
        means, log_std = values.T
        means, log_std = means.T, log_std.T
        dist = Normal(means, torch.exp(log_std))
        return dist

    def _get_log_probs(self, states, actions):
        pi_values = self.pi_model(states)
        dist = self.get_dist(pi_values, states.shape[0])
        log_probs = dist.log_prob(actions)
        return log_probs

    def get_action(self, state):
        state = torch.FloatTensor(state)
        pi_values = self.pi_model(state)
        dist = self.get_dist(pi_values, 1)
        action = dist.sample()
        action = action.numpy().reshape(self.action_dim)
        return action

    def _get_returns(self, rewards, dones):
        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        return returns

    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = self._get_returns(rewards, dones)

        states, actions, returns = map(torch.FloatTensor, [states, actions, returns])

        old_log_probs = self._get_log_probs(states, actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_returns.detach() - self.v_model(b_states)

                b_new_log_probs = self._get_log_probs(b_states, b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                self._update_pi_model(b_advantage, b_ratio)

                self._update_v_model(b_advantage)

    def _update_pi_model(self, advantage, ratio):
        pi_loss_1 = ratio * advantage.detach()
        pi_loss_2 = (
            torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
            * advantage.detach()
        )
        pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

        pi_loss.backward()
        self.pi_optimizer.step()
        self.pi_optimizer.zero_grad()

    def _update_v_model(self, advantage):
        v_loss = torch.mean(advantage**2)

        v_loss.backward()
        self.v_optimizer.step()
        self.v_optimizer.zero_grad()


class PPOAdvantage(PPO):
    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        states, actions, rewards = map(torch.FloatTensor, [states, actions, rewards])

        old_log_probs = self._get_log_probs(states, actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(states.shape[0] - 1)
            for i in range(0, states.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_dones_ = dones[b_idxs].flatten()
                b_idxs = b_idxs[~b_dones_]  # remove terminal states from batch
                b_states = states[b_idxs]
                b_states_ = states[b_idxs + 1]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = (
                    b_rewards.detach()
                    + self.gamma * self.v_model(b_states_).detach()
                    - self.v_model(b_states)
                )

                b_new_log_probs = self._get_log_probs(b_states, b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                self._update_pi_model(b_advantage, b_ratio)

                self._update_v_model(b_advantage)


class PPO_Discrete(nn.Module):
    def __init__(
        self,
        state_dim,
        action_n,
        gamma=0.9,
        batch_size=128,
        epsilon=0.2,
        epoch_n=30,
        pi_lr=1e-4,
        v_lr=5e-4,
    ):
        super().__init__()

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_n),
        )

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

    def get_action(self, state):
        log_probs = self.pi_model(torch.FloatTensor(state))
        dist = Categorical(np.exp(log_probs))
        action = dist.sample()
        action = action.numpy().reshape(1)
        return action

    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns = map(torch.FloatTensor, [states, actions, returns])

        log_probs = self.pi_model(states).T
        log_probs = log_probs.unsqueeze(1)
        dist = Categorical(np.exp(log_probs))
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_returns.detach() - self.v_model(b_states)

                b_log_probs = self.pi_model(b_states).T
                b_log_probs = b_log_probs.unsqueeze(1)
                b_dist = Categorical(np.exp(b_log_probs))
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = (
                    torch.clamp(b_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * b_advantage.detach()
                )
                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage**2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


class PPO_Discrete_2(nn.Module):
    def __init__(
        self,
        state_dim,
        action_n,
        gamma=0.99,
        batch_size=128,
        epsilon=0.2,
        epoch_n=30,
        pi_lr=1e-4,
        v_lr=5e-4,
    ):
        super().__init__()

        self.action_n = action_n

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_n),
            nn.Softmax(dim=-1),
        )

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

    def get_dist(self, pi_values):
        dist = Categorical(pi_values)
        return dist

    def _get_log_probs(self, states, actions):
        pi_values = self.pi_model(states)
        dist = self.get_dist(pi_values)
        log_probs = dist.log_prob(actions)
        return log_probs

    def get_action(self, state):
        state = torch.FloatTensor(state)
        pi_values = self.pi_model(state)
        dist = self.get_dist(pi_values)
        action = dist.sample()
        action = action.numpy()
        return action

    def _get_returns(self, rewards, dones):
        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        return returns

    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = self._get_returns(rewards, dones)

        states, actions, returns = map(torch.FloatTensor, [states, actions, returns])

        old_log_probs = self._get_log_probs(states, actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_returns.detach() - self.v_model(b_states)

                b_new_log_probs = self._get_log_probs(b_states, b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                self._update_pi_model(b_advantage, b_ratio)

                self._update_v_model(b_advantage)

    def _update_pi_model(self, advantage, ratio):
        pi_loss_1 = ratio * advantage.detach()
        pi_loss_2 = (
            torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
            * advantage.detach()
        )
        pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

        pi_loss.backward()
        self.pi_optimizer.step()
        self.pi_optimizer.zero_grad()

    def _update_v_model(self, advantage):
        v_loss = torch.mean(advantage**2)

        v_loss.backward()
        self.v_optimizer.step()
        self.v_optimizer.zero_grad()

class PPO_Discrete_2_Advantage(PPO_Discrete_2):
    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(np.array, [states, actions, rewards, dones])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        states, actions, rewards = map(torch.FloatTensor, [states, actions, rewards])

        old_log_probs = self._get_log_probs(states, actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(states.shape[0] - 1)
            for i in range(0, states.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_dones_ = dones[b_idxs].flatten()
                b_idxs = b_idxs[~b_dones_] # remove terminal states from batch
                b_states = states[b_idxs]
                b_states_ = states[b_idxs + 1]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_rewards.detach() + self.gamma * self.v_model(b_states_).detach() - self.v_model(b_states)

                b_new_log_probs = self._get_log_probs(b_states, b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                self._update_pi_model(b_advantage, b_ratio)

                self._update_v_model(b_advantage)
