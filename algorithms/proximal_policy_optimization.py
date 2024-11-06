import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPO_Base(nn.Module):
    def __init__(
        self,
        pi_model: nn.Module,
        v_model: nn.Module,
        gamma,
        batch_size,
        epsilon,
        epoch_n,
        pi_lr,
        v_lr,
    ):
        super().__init__()
        self.pi_model = pi_model
        self.v_model = v_model
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

    def _update_pi_model(self, advantage: torch.Tensor, ratio: torch.Tensor):
        pi_loss_1 = ratio * advantage.detach()
        pi_loss_2 = (
            torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
            * advantage.detach()
        )
        pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

        pi_loss.backward()
        self.pi_optimizer.step()
        self.pi_optimizer.zero_grad()

    def _update_v_model(self, advantage: torch.Tensor):
        v_loss = torch.mean(advantage**2)

        v_loss.backward()
        self.v_optimizer.step()
        self.v_optimizer.zero_grad()

    def _get_returns(self, rewards: np.ndarray, dones: np.ndarray):
        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        return returns


class PPO_Discrete_Logits_Guided(PPO_Base):
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
        self.action_n = action_n
        self.inner_layer = 256

        pi_model = nn.Sequential(
            nn.Linear(state_dim, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, action_n),
        )

        v_model = nn.Sequential(
            nn.Linear(state_dim, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, 1),
        )

        super().__init__(
            pi_model, v_model, gamma, batch_size, epsilon, epoch_n, pi_lr, v_lr
        )

    def get_dist(self, pi_values, available_actions_mask):
        unavailable_actions_mask = ~available_actions_mask
        pi_values[unavailable_actions_mask] = -torch.inf
        dist = Categorical(logits=pi_values)
        return dist

    def _get_log_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        available_actions_mask: np.ndarray,
    ) -> torch.Tensor:
        pi_values = self.pi_model(states)
        dist = self.get_dist(pi_values, available_actions_mask)
        log_probs = dist.log_prob(actions)
        return log_probs

    def get_action(self, state, info):
        state = torch.FloatTensor(state)
        pi_values = self.pi_model(state)
        available_actions_mask = self._convert_infos([info])
        dist = self.get_dist(pi_values, available_actions_mask[0])
        action = dist.sample()
        action = action.numpy()
        return action

    def _convert_infos(self, infos):
        infos_count = len(infos)
        mask = np.full((infos_count, self.action_n), False)
        for i, info in enumerate(infos):
            available_actions = info["actions"]
            mask[i, available_actions] = True

        return mask

    def _get_advantage(self, returns: torch.Tensor, states: torch.Tensor):
        advantage = returns.detach() - self.v_model(states)
        return advantage

    def _prepare_data(self, states, actions, rewards, dones, infos):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = self._get_returns(rewards, dones)

        states, actions, returns = map(torch.FloatTensor, [states, actions, returns])

        available_actions_mask = self._convert_infos(infos)

        old_log_probs = self._get_log_probs(
            states, actions, available_actions_mask
        ).detach()

        return states, actions, returns, old_log_probs, available_actions_mask

    def _make_policy_step(self, states, actions, advantage, old_log_probs, available_actions_mask):
        new_log_probs = self._get_log_probs(states, actions, available_actions_mask)

        ratio = torch.exp(new_log_probs - old_log_probs)
        self._update_pi_model(advantage, ratio)

        self._update_v_model(advantage)

    def _step(
        self, states, actions, returns, old_log_probs, available_actions_mask
    ):
        advantage = self._get_advantage(returns, states)
        self._make_policy_step(states, actions, advantage, old_log_probs, available_actions_mask)


    def fit(self, states, actions, rewards, dones, infos):
        data = self._prepare_data(states, actions, rewards, dones, infos)
        states, *_ = data
        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(states.shape[0])
            for i in range(0, idxs.shape[0] // self.batch_size):
                batch_indexes = idxs[i * self.batch_size : (i + 1) * self.batch_size]
                batch_data = map(lambda array: array[batch_indexes], data)
                self._step(*batch_data)


class PPO_Discrete_Logits_Guided_Advantage(PPO_Discrete_Logits_Guided):
    def _get_advantage(
        self, rewards: torch.Tensor, states: torch.Tensor, next_states: torch.Tensor
    ):
        advantage = (
            rewards.detach()
            + self.gamma * self.v_model(next_states).detach()
            - self.v_model(states)
        )
        return advantage

    def _prepare_data(self, states, actions, rewards, dones, infos):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards = rewards.reshape(-1, 1)
        undones = ~dones.flatten()

        states, actions, rewards = map(torch.FloatTensor, [states, actions, rewards])

        available_actions_mask = self._convert_infos(infos)

        next_states = states[1:]
        states = states[undones]  # remove terminal states from batch
        actions = actions[undones]
        rewards = rewards[undones]
        next_states = next_states[undones[:-1]]
        available_actions_mask: np.ndarray = available_actions_mask[undones]

        old_log_probs = self._get_log_probs(
            states, actions, available_actions_mask
        ).detach()

        return (
            states,
            next_states,
            actions,
            rewards,
            old_log_probs,
            available_actions_mask,
        )

    def _step(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        available_actions_mask: np.ndarray,
    ):
        advantage = self._get_advantage(rewards, states, next_states)
        self._make_policy_step(states, actions, advantage, old_log_probs, available_actions_mask)
