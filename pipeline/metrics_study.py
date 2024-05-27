from environment.environment import ProductOwnerEnv
from .base_study import BaseStudy


import torch


from typing import List


class MetricsStudy(BaseStudy):
    def __init__(self, env: ProductOwnerEnv, agent, trajectory_max_len) -> None:
        super().__init__(env, agent, trajectory_max_len)
        self.rewards_log: List[int] = []
        self.discounted_rewards_log: List[float] = []
        self.q_value_log: List[int] = []

    def play_trajectory(self, init_state, init_info, init_discount=1):
        with torch.no_grad():
            state = torch.tensor(init_state).to(self.agent.device)
            q_values: torch.Tensor = self.agent.q_function.predict(state)
        estimates = q_values.max().detach().cpu().numpy()
        self.q_value_log.append(estimates)

        reward, discounted_reward = super().play_trajectory(init_state, init_info, init_discount)
        self.rewards_log.append(reward)
        self.discounted_rewards_log.append(discounted_reward)
        return reward, discounted_reward
