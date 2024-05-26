import math
from .metrics_study import MetricsStudy
from .study_agent import save_dqn_agent


import datetime
import os
import sys
import logging
from typing import List, Tuple, Optional


class LoggingStudy(MetricsStudy):
    SAVE_MEMORY = False

    def __init__(
        self,
        env,
        agent,
        trajectory_max_len,
        save_rate: Optional[int] = None,
        log_level=logging.INFO,
    ) -> None:
        super().__init__(env, agent, trajectory_max_len)
        self.episode = 0
        self.sprints_log: List[int] = []
        self.loss_log: List[float] = []
        self.time_log: List[datetime.datetime] = []
        self.save_rate = save_rate
        self.logger = self._get_logger(log_level)

    def _get_logger(self, log_level):
        logger = logging.getLogger(f'{datetime.datetime.now()}')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

        return logger

    def _log_before_action(self, action):
        message = None
        if action == 1:
            cards_count = len(self.env.game.userstories.release)
            message = f'decompose {cards_count} cards'
        if action == 2:
            released_count = len(self.env.game.completed_us)
            message = f'release {released_count} user stories'
        if action == 3:
            message = 'buy robot'
        if action == 4:
            message = 'buy room'
        if action == 5:
            message = 'buy statistical research'
        if action == 6:
            message = 'buy user survey'

        if message:
            self.logger.debug(message)

    def _log_after_action(self, action):
        message = None
        if action == 0:
            message = 'start sprint'
            money = self.env.game.context.get_money()
            loyalty = self.env.game.context.get_loyalty()
            customers = self.env.game.context.customers
            current_sprint = self.env.game.context.current_sprint
            message += f" {current_sprint}: {money / 1e5:2.4f}, {loyalty:2.4f}, {customers:3.4f}"
        if action >= 7:
            message = 'move card'
            current_hours = self.env.game.backlog.calculate_hours_sum()
            max_sprint_hours = self.env.game.backlog.get_max_hours()
            message += f": {current_hours}/{max_sprint_hours}"

        if message:
            self.logger.debug(message)

    def fit_agent(self, state, action, reward, done, next_state, next_info):
        loss = super().fit_agent(state, action, reward, done, next_state, next_info)
        self._log_after_action(action)
        self.loss_log.append(loss)
        return loss

    def play_trajectory(self, init_state, init_info) -> float:
        reward = super().play_trajectory(init_state, init_info)
        self._log_trajectory_end(reward)

        return reward

    def _log_trajectory_end(self, reward):
        sprint_n = self.env.game.context.current_sprint

        self.sprints_log.append(sprint_n)

        credit = self.env.game.context.credit

        termination = "none"
        if not self.env.game.context.is_new_game:
            termination = 'tutorial'
        if self.env.game.context.credit == 0:
            termination = 'credit paid'
        if self.env.game.context.is_victory:
            termination = "victory"
        if self.env.game.context.is_loss:
            termination = "lose"

        message = (
                f"episode: {(self.episode + 1):03d}\t"
                + f"total_reward: {reward:.2f}\t"
                + f"sprint_n: {sprint_n:02d}\t"
                + f"credit: {credit: 6d}\t"
                + f"termination: {termination}\t"
        )
        self.logger.info(message)
        self.episode += 1

    def _choose_action(self, action, inner_sprint_action_count) -> Tuple[int, int]:
        result = super()._choose_action(action, inner_sprint_action_count)
        chosen_action, _ = result
        if action != chosen_action and chosen_action == 0:
            self.logger.debug('enforced next sprint')
        self._log_before_action(action)
        return result

    def study_agent(self, episode_n):
        agent_name = type(self.agent).__name__
        env_name = type(self.env).__name__
        epoch_n = self.define_epoch_count_and_save_rate(episode_n)

        os.makedirs(agent_name, exist_ok=True)

        for epoch in range(epoch_n):
            path = f"{agent_name}/model_{epoch}_{env_name}.pt"
            super().study_agent(self.save_rate)
            self.save_model(path, agent_name, env_name, epoch)

    def define_epoch_count_and_save_rate(self, episode_n) -> int:
        if self.save_rate is None:
            epoch_n = 1
            self.save_rate = episode_n
        else:
            epoch_n = math.ceil(episode_n / self.save_rate)

        return epoch_n

    def save_model(self, path, agent_name, env_name, epoch):
        memory = self.agent.memory
        if not self.SAVE_MEMORY:
            self.agent.memory = []
        save_dqn_agent(self.agent, path=path)
        self.agent.memory = memory
        with open(f"{agent_name}/rewards_{epoch}_{env_name}.txt", mode="w") as f:
            f.write(repr(self.rewards_log))
        with open(f"{agent_name}/estimates_{epoch}_{env_name}.txt", mode="w") as f:
            f.write(repr(self.q_value_log))
        with open(f"{agent_name}/sprints_{epoch}_{env_name}.txt", mode="w") as f:
            f.write(repr(self.sprints_log))
        with open(f"{agent_name}/loss_{epoch}_{env_name}.txt", mode="w") as f:
            f.write(repr(self.loss_log))
