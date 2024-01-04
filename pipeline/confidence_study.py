from environment import ProductOwnerEnv

from typing import List

from .metrics_study import MetricsStudy


class ConfidenceStudy(MetricsStudy):
    def __init__(self, env: ProductOwnerEnv, agent_generator, trajectory_max_len) -> None:
        super().__init__(env, agent_generator(), trajectory_max_len)
        self.agent_generator = agent_generator
        self.rewards_logs: List[List[int]] = []
        self.q_value_logs: List[List[int]] = []
    
    def study_agents(self, episode_n, repeat_count):
        for i in range(repeat_count):
            self.agent = self.agent_generator()
            super().study_agent(episode_n)
            print(f'Complete study #{i} with reward {self.rewards_log[-1]}')
            self.rewards_logs.append(self.rewards_log)
            self.q_value_logs.append(self.q_value_log)
            self.rewards_log = []
            self.q_value_log = []
