from .ValueFunction import ValueFunction
from .q_function import QFunction
from .PolicyFunction import PolicyFunction
from .deep_q_networks import DQN, HardTargetDQN, SoftTargetDQN, DoubleDQN
from .proximal_policy_optimization import (
    PPO_Discrete_Logits_Guided,
    PPO_Discrete_Logits_Guided_Advantage,
)


class DqnAgentsFactory:
    def __init__(self):
        self.gamma = 0.9
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.epsilon_decrease = 1e-4
        self.epsilon_min = 0.01
        self.target_update = 100
        self.tau = 1e-3
        self.q_function_embeding_size = 256

    def create_dqn(self, state_dim, action_n):
        q_function = QFunction(state_dim, action_n, self.q_function_embeding_size)
        agent = DQN(
            q_function,
            gamma=self.gamma,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            epsilon_decrease=self.epsilon_decrease,
            epsilon_min=self.epsilon_min,
        )

        return agent

    def create_hard_target_dqn(self, state_dim, action_n):
        q_function = QFunction(state_dim, action_n, self.q_function_embeding_size)
        agent = HardTargetDQN(
            q_function,
            gamma=self.gamma,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            target_update=self.target_update,
            epsilon_decrease=self.epsilon_decrease,
            epsilon_min=self.epsilon_min,
        )
        return agent

    def create_soft_target_dqn(self, state_dim, action_n):
        q_function = QFunction(state_dim, action_n, self.q_function_embeding_size)
        agent = SoftTargetDQN(
            q_function,
            gamma=self.gamma,
            tau=self.tau,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            epsilon_decrease=self.epsilon_decrease,
            epsilon_min=self.epsilon_min,
        )
        return agent

    def create_ddqn(self, state_dim, action_n):
        q_function = QFunction(state_dim, action_n, self.q_function_embeding_size)
        agent = DoubleDQN(
            q_function,
            gamma=self.gamma,
            tau=self.tau,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            epsilon_decrease=self.epsilon_decrease,
            epsilon_min=self.epsilon_min,
        )
        return agent


class PPOAgentsFactory:
    def __init__(self):
        self.gamma = 0.9
        self.batch_size = 128
        self.epsilon = 0.2
        self.epoch_n = 5
        self.pi_lr = 1e-4
        self.v_lr = 5e-4
        self.inner_layer = 256

    def create_ppo_discrete_logits_guided(self, state_dim, action_n):
        policy_function = PolicyFunction(
            state_dim, action_n, inner_layer=self.inner_layer
        )
        value_function = ValueFunction(state_dim, inner_layer=self.inner_layer)

        agent = PPO_Discrete_Logits_Guided(
            state_dim,
            action_n,
            pi_model=policy_function,
            v_model=value_function,
            gamma=self.gamma,
            batch_size=self.batch_size,
            epsilon=self.epsilon,
            epoch_n=self.epoch_n,
            pi_lr=self.pi_lr,
            v_lr=self.v_lr,
        )

        return agent

    def create_ppo_discrete_logits_guided_advantage(self, state_dim, action_n):
        policy_function = PolicyFunction(
            state_dim, action_n, inner_layer=self.inner_layer
        )
        value_function = ValueFunction(state_dim, inner_layer=self.inner_layer)

        agent = PPO_Discrete_Logits_Guided_Advantage(
            state_dim,
            action_n,
            pi_model=policy_function,
            v_model=value_function,
            gamma=self.gamma,
            batch_size=self.batch_size,
            epsilon=self.epsilon,
            epoch_n=self.epoch_n,
            pi_lr=self.pi_lr,
            v_lr=self.v_lr,
        )

        return agent
