from .q_function import QFunction
from .deep_q_networks import DQN, HardTargetDQN, SoftTargetDQN, DoubleDQN


class DqnAgentsFactory:
    def __init__(self, q_function_embeding_size=512):
        self.gamma = 0.9
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.epsilon_decrease = 1e-4
        self.epsilon_min = 0.01
        self.target_update = 100
        self.tau = 1e-3
        self.q_function_embeding_size = q_function_embeding_size

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
