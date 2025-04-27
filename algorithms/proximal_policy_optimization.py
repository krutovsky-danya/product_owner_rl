import numpy as np
import torch
import torch.nn as nn

from operator import itemgetter
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

from .PolicyFunction import PolicyFunction
from .ValueFunction import ValueFunction


class PPO_Base(nn.Module):
    def __init__(
        self,
        pi_model: PolicyFunction,
        v_model: ValueFunction,
        gamma,
        batch_size,
        epsilon,
        epoch_n,
        pi_lr,
        v_lr,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pi_model: PolicyFunction = pi_model.to(self.device)
        self.v_model: ValueFunction = v_model.to(self.device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def fit(self, states, actions, rewards, dones, infos):
        pass

    def store_transition(self, state, action, reward, done, info):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def update_policy(self):
        states, actions, rewards, dones, infos = (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.infos,
        )
        self.fit(states, actions, rewards, dones, infos)

        # Clear the stored transitions after training
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.infos.clear()

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

    @torch.no_grad()
    def get_value(self, state, info):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        value = self.v_model.forward(state)
        return value


class PPO_Discrete_Logits_Guided(PPO_Base):
    def __init__(
        self,
        state_dim,
        action_n,
        pi_model: PolicyFunction = None,
        v_model: ValueFunction = None,
        gamma=0.99,
        batch_size=128,
        epsilon=0.2,
        epoch_n=30,
        pi_lr=1e-4,
        v_lr=5e-4,
    ):
        self.action_n = action_n

        super().__init__(
            pi_model, v_model, gamma, batch_size, epsilon, epoch_n, pi_lr, v_lr
        )

    def _get_log_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        available_actions_mask: np.ndarray,
    ) -> torch.Tensor:
        pi_values = self.pi_model.forward_guided(states, available_actions_mask)
        dist = Categorical(probs=pi_values)
        log_probs = dist.log_prob(actions)
        return log_probs

    def get_action(self, state, info):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        available_actions_mask = self._convert_infos([info])
        pi_values = self.pi_model.forward_guided(state, available_actions_mask)
        dist = Categorical(probs=pi_values)
        action = dist.sample()
        action = action.item()
        return action

    def _convert_infos(self, infos) -> torch.BoolTensor:
        """Converts infos to a mask of available actions."""
        guides = map(itemgetter("actions"), infos)
        mask = torch.tensor(list(guides), dtype=torch.bool, device=self.device)
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

        to_float_tensor = lambda x: torch.tensor(
            x, dtype=torch.float, device=self.device
        )
        states, actions, returns = map(to_float_tensor, [states, actions, returns])

        available_actions_mask = self._convert_infos(infos)

        old_log_probs = self._get_log_probs(
            states, actions, available_actions_mask
        ).detach()

        return states, actions, returns, old_log_probs, available_actions_mask

    def _make_policy_step(
        self, states, actions, advantage, old_log_probs, available_actions_mask
    ):
        new_log_probs = self._get_log_probs(states, actions, available_actions_mask)

        ratio = torch.exp(new_log_probs - old_log_probs)
        self._update_pi_model(advantage, ratio)

        self._update_v_model(advantage)

    def _step(self, states, actions, returns, old_log_probs, available_actions_mask):
        advantage = self._get_advantage(returns, states)
        self._make_policy_step(
            states, actions, advantage, old_log_probs, available_actions_mask
        )

    def fit(self, states, actions, rewards, dones, infos):
        data = self._prepare_data(states, actions, rewards, dones, infos)

        dataset = TensorDataset(*data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch_n):
            for batch_data in dataloader:
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

        to_float_tensor = lambda x: torch.tensor(
            x, dtype=torch.float, device=self.device
        )
        states, actions, rewards = map(to_float_tensor, [states, actions, rewards])

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
        self._make_policy_step(
            states, actions, advantage, old_log_probs, available_actions_mask
        )
