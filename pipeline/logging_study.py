from .metrics_study import MetricsStudy
from .study_agent import save_dqn_agent


import datetime
import os
from typing import List


class LoggingStudy(MetricsStudy):
    SAVE_MEMORY = True

    def __init__(self, env, agent, trajectory_max_len, save_rate=1000) -> None:
        super().__init__(env, agent, trajectory_max_len)
        self.episode = 0
        self.sprints_log: List[int] = []
        self.loss_log: List[float] = []
        self.time_log: List[datetime.datetime] = []
        self.save_rate = save_rate

    def fit_agent(self, state, action, reward, done, next_state):
        loss = super().fit_agent(state, action, reward, done, next_state)
        self.loss_log.append(loss)
        return loss

    def play_trajectory(self, init_state):
        reward = super().play_trajectory(init_state)
        sprint_n = self.env.game.context.current_sprint

        self.sprints_log.append(sprint_n)

        credit = self.env.game.context.credit

        termination = "none"
        if self.env.game.context.is_victory:
            termination = "victory"
        if self.env.game.context.is_loss:
            termination = "lose"

        message = (
            f"\nepisode: {self.episode:03d}\t"
            + f"total_reward: {reward:.2f}\t"
            + f"sprint_n: {sprint_n:02d}\t"
            + f"credit: {credit: 6d}\t"
            + f"termination: {termination}\t"
        )

        print(message)
        self.episode += 1

    def study_agent(self, episode_n):
        agent_name = type(self.agent).__name__
        epoch_n = (episode_n + self.save_rate - 1) // self.save_rate

        os.makedirs(agent_name, exist_ok=True)

        for epoch in range(epoch_n):
            path = f"{agent_name}/model_{epoch}.pt"
            super().study_agent(self.save_rate)
            memory = self.agent.memory
            if not self.SAVE_MEMORY:
                self.agent.memory = []
            save_dqn_agent(self.agent, path=path)
            self.agent.memory = memory
            with open(f"{agent_name}/rewards_{epoch}.txt", mode="w") as f:
                f.write(repr(self.rewards_log))
            with open(f"{agent_name}/estimates_{epoch}.txt", mode="w") as f:
                f.write(repr(self.q_value_log))
            with open(f"{agent_name}/sprints_{epoch}.txt", mode="w") as f:
                f.write(repr(self.sprints_log))
