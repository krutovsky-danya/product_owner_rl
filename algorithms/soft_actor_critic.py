import torch
from torch import nn
import numpy as np
import random


def get_nan_safe_log(values):
    # small_value = values <= 0.0
    # small_value = small_value.float() * 1e-8
    small_value = 1e-8
    log_values = torch.log(values + small_value)
    return log_values


class BaseNeuralFunction(nn.Module):
    def __init__(self, state_dim, action_n, inner_layer=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.inner_layer = inner_layer

        self.network = nn.Sequential(
            nn.Linear(state_dim, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, action_n)
        )

    def forward(self, states):
        return self.network(states)

    def predict(self, state):
        self.network.eval()
        result = self.network(state.unsqueeze(0))
        self.network.train(True)
        return result


class QFunction(BaseNeuralFunction):
    def __init__(self, state_dim, action_n, inner_layer=128):
        super().__init__(state_dim, action_n, inner_layer)

    def get_target_copy(self, device):
        target_q_function: nn.Module = QFunction(self.state_dim, self.action_n, self.inner_layer)
        target_q_function = target_q_function.to(device)

        state_dict = self.state_dict()
        target_q_function.load_state_dict(state_dict)

        for param in target_q_function.parameters():
            param.requires_grad = False

        return target_q_function


class PolicyFunction(BaseNeuralFunction):
    def __init__(self, state_dim, action_n, inner_layer=128):
        super().__init__(state_dim, action_n, inner_layer)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, states):
        logits = super().forward(states)
        res = self.softmax(logits)
        if torch.any(res < 0).item():
            print(f"hold on! wain a sec. value {res[res < 0]}")
        return self.softmax(logits)

    def predict(self, state):
        results = super().predict(state)
        res = self.softmax(results)
        if torch.any(res < 0).item():
            print(f"hold on! wain a sec. value {res[res < 0]}")
        return self.softmax(results)


class SAC(nn.Module):
    def __init__(
            self,
            state_dim,
            action_n,
            gamma=0.99,
            alpha=0.001,
            batch_size=64,
            policy_learning_rate=1e-4,
            q_learning_rate=1e-3,
            tau=0.01
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_function_1: QFunction = QFunction(self.state_dim,
                                                 self.action_n).to(device=self._device)
        self.q_function_2: QFunction = QFunction(self.state_dim,
                                                 self.action_n).to(device=self._device)

        self.q_optimizer_1 = torch.optim.Adam(self.q_function_1.parameters(), lr=q_learning_rate)
        self.q_optimizer_2 = torch.optim.Adam(self.q_function_2.parameters(), lr=q_learning_rate)

        self.policy_function: PolicyFunction = PolicyFunction(self.state_dim,
                                                              self.action_n).to(device=self._device)
        self.policy_optimizer = torch.optim.Adam(self.policy_function.parameters(),
                                                 lr=policy_learning_rate)

        self.target_q_function_1 = self.q_function_1.get_target_copy(self._device)
        self.target_q_function_2 = self.q_function_2.get_target_copy(self._device)

        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory = []
        self.tau = tau
        self.loss = []
        self.loss_q1 = []
        self.loss_q2 = []

    @torch.no_grad()
    def get_action(self, state, info):
        mask = info["actions"]
        # todo mask
        state = torch.FloatTensor(state).to(device=self._device)
        probs = self.policy_function.predict(state).squeeze().cpu()
        # had an error with probs sum not being equal to 1 due to problems with accuracy
        # and inner implicit casting to float64 from float32
        probs = np.asarray(probs).astype("float64")
        probs /= np.sum(probs)
        action = np.random.choice(self.action_n, p=probs)
        return action

    def get_padded_to_action_n(self, guide: torch.Tensor):
        # todo if mask will be True/False we don't need this method
        return torch.nn.functional.pad(guide,
                                       pad=(0, self.policy_function.action_n - guide.numel()),
                                       mode="constant",
                                       value=guide[0].item())

    def fit(self, state, action, reward, done, next_state, next_info):
        if not self.training:
            return
        next_guide = torch.tensor(next_info["actions"])
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

        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        dones = dones.to(self._device)
        next_states = next_states.to(self._device)
        next_guides = next_guides.to(self._device)

        self._gradient_step(states, actions, rewards, dones, next_states, next_guides)

    def _gradient_step(self, states, actions, rewards, dones, next_states, next_guides):
        # todo guides
        loss_q_1, loss_q_2 = self._q_gradient_step(states, actions, rewards, dones, next_states, next_guides)
        loss_policy = self._policy_gradient_step(states, actions, rewards, dones, next_states, next_guides)

        self.loss.append(loss_policy)
        self.loss_q1.append(loss_q_1)
        self.loss_q2.append(loss_q_2)
        self._update_targets()
        return loss_policy, loss_q_1, loss_q_2

    def _q_gradient_step(self, states, actions, rewards, dones, next_states, next_guides):
        # todo guides
        q_target_next_1 = self.target_q_function_1(next_states)
        q_target_next_2 = self.target_q_function_2(next_states)

        next_action_probs = self.policy_function(next_states)
        next_action_log_probs = get_nan_safe_log(next_action_probs)
        entropy = self.alpha * next_action_log_probs

        v_target_next_value_1 = torch.sum(next_action_probs * (q_target_next_1 - entropy), dim=1)
        v_target_next_value_2 = torch.sum(next_action_probs * (q_target_next_2 - entropy), dim=1)
        min_v_value = torch.min(v_target_next_value_1, v_target_next_value_2)
        targets = rewards + self.gamma * (1 - dones) * min_v_value
        targets = targets.detach()

        q_values_1 = self.q_function_1(states)[torch.arange(self.batch_size), actions]
        q_values_2 = self.q_function_2(states)[torch.arange(self.batch_size), actions]

        loss_q_1 = torch.mean((q_values_1 - targets) ** 2)
        loss_q_2 = torch.mean((q_values_2 - targets) ** 2)

        self._take_optimization_step(self.q_optimizer_1, loss_q_1)
        self._take_optimization_step(self.q_optimizer_2, loss_q_2)

        return loss_q_1, loss_q_2

    def _policy_gradient_step(self, states, actions, rewards, dones, next_states, next_guides):
        q_value_1 = self.q_function_1(states).detach()
        q_value_2 = self.q_function_2(states).detach()

        min_q_value = torch.min(q_value_1, q_value_2)

        action_probs = self.policy_function(states)
        action_log_probs = get_nan_safe_log(action_probs)
        entropy = self.alpha * action_log_probs

        loss_policy = torch.mean(torch.sum(action_probs * (entropy - min_q_value), dim=1))
        self._take_optimization_step(self.policy_optimizer, loss_policy)

        return loss_policy

    def _take_optimization_step(self, optimizer, loss):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _update_targets(self):
        self._update_target(self.q_function_1, self.target_q_function_1)
        self._update_target(self.q_function_2, self.target_q_function_2)

    def _update_target(self, q_function: QFunction, target_q_function: QFunction):
        # theta' = tau * theta + (1 - tau) * theta'
        target_dict = target_q_function.state_dict()
        for name, param in q_function.named_parameters():
            target_dict[name] = (
                self.tau * param.data + (1 - self.tau) * target_dict[name]
            )
        target_q_function.load_state_dict(target_dict)
