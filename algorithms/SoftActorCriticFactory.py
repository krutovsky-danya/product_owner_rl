from .soft_actor_critic import SoftActorCritic, SACWithLearnedTemperature


class SoftActorCriticFactory:
    def __init__(self):
        self.alpha = 0.001
        self.gamma = 0.9
        self.tau = 1e-3
        self.batch_size = 64
        self.policy_lr = 1e-4
        self.q_function_lr = 1e-3
        self.alpha_lr = 1e-4
        self.embedding_size = 256
        self.entropy_target = 1

    def create_soft_actor_critic(self, state_dim, action_n):
        agent = SoftActorCritic(
            state_dim=state_dim,
            action_n=action_n,
            gamma=self.gamma,
            alpha=self.alpha,
            batch_size=self.batch_size,
            policy_learning_rate=self.policy_lr,
            q_learning_rate=self.q_function_lr,
            tau=self.tau,
        )
        return agent

    def create_soft_actor_critic_with_entropy(self, state_dim, action_n):
        agent = SACWithLearnedTemperature(
            state_dim=state_dim,
            action_n=action_n,
            gamma=self.gamma,
            batch_size=self.batch_size,
            policy_learning_rate=self.policy_lr,
            q_learning_rate=self.q_function_lr,
            tau=self.tau,
            entropy_target=self.entropy_target,
            alpha_learning_rate=self.alpha_lr,
        )
        return agent

    def get_agents(self):
        return [
            self.create_soft_actor_critic,
            self.create_soft_actor_critic_with_entropy,
        ]
