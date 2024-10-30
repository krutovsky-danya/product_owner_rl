import torch
from torch import nn
import numpy as np
import random


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
            nn.Linear(self.inner_layer, self.inner_layer),
            nn.ReLU(),
            nn.Linear(self.inner_layer, action_n)
        )

    def forward(self, states):
        return self.network(states)

    @torch.no_grad()
    def predict(self, state):
        self.network.eval()
        result = self.network(state.unsqueeze(0))
        self.network.train(True)
        return result


class QFunction(BaseNeuralFunction):
    def __init__(self, state_dim, action_n, inner_layer=256):
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

    def forward_guided(self, states, guides):
        logits: torch.Tensor = super().forward(states)
        # masked_fill is out-of-place version of masked_fill_
        # (in code it usually looks like this: tensor[mask] = value
        # where mask is a Tensor of booleans)
        logits = logits.masked_fill(~guides, -torch.inf)
        return self.softmax(logits)

    def get_normalized_guided_logits(self, states, guides):
        logits: torch.Tensor = super().forward(states)
        logits = logits.masked_fill(~guides, -torch.inf)
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return logits

    @torch.no_grad()
    def predict_guided(self, state, guide):
        logits = super().predict(state)
        guide = guide.unsqueeze(dim=0)
        logits[~guide] = -torch.inf
        return self.softmax(logits)


def _take_optimization_step(optimizer, loss):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _convert_loss(loss):
    return loss.cpu().detach().numpy()


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_function_1: QFunction = QFunction(self.state_dim,
                                                 self.action_n).to(device=self.device)
        self.q_function_2: QFunction = QFunction(self.state_dim,
                                                 self.action_n).to(device=self.device)

        self.q_optimizer_1 = torch.optim.Adam(self.q_function_1.parameters(), lr=q_learning_rate)
        self.q_optimizer_2 = torch.optim.Adam(self.q_function_2.parameters(), lr=q_learning_rate)

        self.policy_function: PolicyFunction = PolicyFunction(self.state_dim,
                                                              self.action_n).to(device=self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_function.parameters(),
                                                 lr=policy_learning_rate)

        self.target_q_function_1 = self.q_function_1.get_target_copy(self.device)
        self.target_q_function_2 = self.q_function_2.get_target_copy(self.device)

        self.gamma = gamma
        self.alpha = torch.full([1], alpha, requires_grad=True, device=self.device)
        self.batch_size = batch_size
        self.memory = []
        self.tau = tau

        self.small_const = torch.Tensor([1e-8]).to(self.device)

    @torch.no_grad()
    def get_action(self, state, info):
        mask = info["actions"]
        mask = self._convert_info(torch.tensor(mask))
        state = torch.FloatTensor(state).to(device=self.device)
        masked_probs = self.policy_function.predict_guided(state, mask).squeeze().cpu()
        if self.training:
            # had an error with probs sum not being equal to 1 due to problems with accuracy
            # and inner implicit casting from float32 to float64
            np_probs = np.asarray(masked_probs).astype("float64")
            np_probs /= np.sum(np_probs)
            action = np.random.choice(self.action_n, p=np_probs)
        else:
            action = torch.argmax(masked_probs).item()
        return action

    def _convert_info(self, guide: torch.Tensor):
        converted = torch.zeros(self.action_n, dtype=torch.bool)
        converted[guide] = True
        return converted

    def _sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, guides, actions, rewards, dones, next_states, next_guides = map(
            torch.stack, list(zip(*batch))
        )

        states = states.to(self.device)
        guides = guides.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        next_guides = next_guides.to(self.device)

        return states, guides, actions, rewards, dones, next_states, next_guides

    def fit(self, state, info, action, reward, done, next_state, next_info):
        if not self.training:
            return
        guide = torch.tensor(info["actions"])
        next_guide = torch.tensor(next_info["actions"])
        self.memory.append(
            [
                torch.tensor(state),
                self._convert_info(guide),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward),
                torch.tensor(int(done)),
                torch.tensor(next_state),
                self._convert_info(next_guide)
            ]
        )

        if len(self.memory) < self.batch_size:
            return

        batch = self._sample_batch()

        loss = self._gradient_step(*batch)

        self._update_targets()

        return loss

    def _gradient_step(self, states, guides, actions, rewards, dones, next_states, next_guides):
        loss_q = self._q_gradient_step(states, actions, rewards, dones, next_states, next_guides)
        loss_policy = self._policy_gradient_step(states, guides)

        losses = (loss_policy, *loss_q)

        losses_for_log = list(map(_convert_loss, losses))

        return losses_for_log

    def _get_probs_and_log_probs(self, states, guides):
        probs = self.policy_function.forward_guided(states, guides)
        log_probs = self.policy_function.get_normalized_guided_logits(states, guides)
        # since probs[~guides] values are supposed to be equal to 0.0
        # as long as we multiply probs and log_probs
        # we do not care what is the value of log_probs[~guides]
        # and also encountering -torch.inf during backward gives us nan
        # so we'll set log_probs[~guides] to 0.0
        log_probs = log_probs.masked_fill(~guides, 0.0)

        return probs, log_probs

    def _get_probs_and_entropy(self, states, guides):
        probs, log_probs = self._get_probs_and_log_probs(states, guides)

        entropy = probs * self.alpha.detach() * log_probs

        return probs, entropy

    def _q_gradient_step(self, states, actions, rewards, dones, next_states, next_guides):
        next_q_target_1 = self.target_q_function_1.forward(next_states)
        next_q_target_2 = self.target_q_function_2.forward(next_states)
        next_min_q_target = torch.min(next_q_target_1, next_q_target_2)

        next_action_probs, entropy = self._get_probs_and_entropy(next_states, next_guides)

        min_v_value = torch.sum(next_action_probs * next_min_q_target - entropy, dim=1)
        targets = rewards + self.gamma * (1 - dones) * min_v_value
        targets = targets.detach()

        q_values_1 = self.q_function_1(states)[torch.arange(self.batch_size), actions]
        q_values_2 = self.q_function_2(states)[torch.arange(self.batch_size), actions]

        loss_q_1 = torch.mean((q_values_1 - targets) ** 2)
        loss_q_2 = torch.mean((q_values_2 - targets) ** 2)

        _take_optimization_step(self.q_optimizer_1, loss_q_1)
        _take_optimization_step(self.q_optimizer_2, loss_q_2)

        return loss_q_1, loss_q_2

    def _policy_gradient_step(self, states, guides):
        # учитывая, что loss_policy ─ это сумма с учетом вероятностей действий action_probs
        # (которые обнулены на недоступных действиях и отнормированы softmax по доступным),
        # а также то, что градиент в данном случае не будет течь через значения q функции,
        # не важно, какие значения принимают q_value на недоступных действиях
        q_value_1 = self.q_function_1.forward(states).detach()
        q_value_2 = self.q_function_2.forward(states).detach()

        min_q_value = torch.min(q_value_1, q_value_2)

        action_probs, entropy = self._get_probs_and_entropy(states, guides)

        loss_policy = torch.mean(torch.sum(entropy - action_probs * min_q_value, dim=1))
        _take_optimization_step(self.policy_optimizer, loss_policy)

        return loss_policy

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

    def train(self, mode: bool = True, epsilon: float = 0):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)


class SACWithLearnedTemperature(SAC):
    def __init__(
            self,
            state_dim,
            action_n,
            gamma=0.99,
            entropy_target=1.0,
            batch_size=64,
            policy_learning_rate=1e-4,
            q_learning_rate=1e-3,
            alpha_learning_rate=1e-3,
            tau=0.01
    ):
        alpha = 1.0
        super().__init__(state_dim, action_n, gamma, alpha, batch_size,
                         policy_learning_rate, q_learning_rate, tau)

        self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=alpha_learning_rate)
        self.entropy_target = entropy_target

    def _gradient_step(self, states, guides, actions, rewards, dones, next_states, next_guides):
        losses = super()._gradient_step(states, guides, actions, rewards, dones,
                                        next_states, next_guides)
        loss_alpha = self._alpha_gradient_step(states, guides)

        return (*losses, _convert_loss(loss_alpha))

    def _alpha_gradient_step(self, states, guides):
        action_probs, action_log_probs = self._get_probs_and_log_probs(states, guides)
        action_probs = action_probs.detach()
        action_log_probs = action_log_probs.detach()

        pre_loss_alpha = action_probs * (-self.alpha * (action_log_probs + self.entropy_target))

        loss_alpha = torch.mean(torch.sum(pre_loss_alpha, dim=1))
        _take_optimization_step(self.alpha_optimizer, loss_alpha)

        return loss_alpha
