import math
from .metrics_study import MetricsStudy
from .study_agent import save_dqn_agent


import datetime
import os
import sys
import logging
from typing import List, Tuple, Optional
from environment.Action import Action
from pipeline.logging_utils import KeyLogState, get_log_entry_creator

loggers = {}

class LoggingStudy(MetricsStudy):
    REWARDS_LOG_KEY = "rewards"
    DISCOUNTED_REWARDS_LOG_KEY = "discounted_rewards"
    ESTIMATES_LOG_KEY = "estimates"
    SPRINTS_LOG_KEY = "sprints"
    LOSS_LOG_KEY = "loss"

    def __init__(
        self,
        env,
        agent,
        trajectory_max_len,
        save_rate: Optional[int] = None,
        save_memory=False,
        base_epoch_log_state=KeyLogState.FULL_LOG,
        base_end_log_state=KeyLogState.FULL_LOG,
        log_level=logging.INFO,
    ) -> None:
        super().__init__(env, agent, trajectory_max_len)
        self.episode = 0
        self.sprints_log: List[int] = []
        self.loss_log: List[float] = []
        self.time_log: List[datetime.datetime] = []
        self.save_rate = save_rate
        self.save_memory = save_memory
        self.logger = self._get_logger(agent.__class__.__name__, log_level)
        create_log_entry = get_log_entry_creator(base_epoch_log_state, base_end_log_state)
        self._logs = {
            self.REWARDS_LOG_KEY: create_log_entry(self.rewards_log),
            self.DISCOUNTED_REWARDS_LOG_KEY: create_log_entry(self.discounted_rewards_log),
            self.ESTIMATES_LOG_KEY: create_log_entry(self.q_value_log),
            self.SPRINTS_LOG_KEY: create_log_entry(self.sprints_log),
            self.LOSS_LOG_KEY: create_log_entry(self.loss_log),
        }

    def _get_logger(self, name, log_level):
        if name in loggers:
            return loggers.get(name)
        
        logger = logging.getLogger(name)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(name)s %(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

        loggers[name] = logger

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
        if action == Action.START_SPRINT:
            message = 'start sprint'
            money = self.env.game.context.get_money()
            loyalty = self.env.game.context.get_loyalty()
            customers = self.env.game.context.customers
            current_sprint = self.env.game.context.current_sprint
            message += f" {current_sprint}: Money = {money / 1e6:2.4f}, Loyalty = {loyalty:2.4f},  Customers = {customers:3.4f}"
        if action >= 7:
            message = 'move card'
            current_hours = self.env.game.backlog.calculate_hours_sum()
            max_sprint_hours = self.env.game.backlog.get_max_hours()
            message += f": {current_hours}/{max_sprint_hours}"

        if message:
            self.logger.debug(message)

    def fit_agent(self, state, info, action, reward, done, next_state, next_info):
        loss = super().fit_agent(state, info, action, reward, done, next_state, next_info)
        self._log_after_action(action)
        self.loss_log.append(loss)
        return loss

    def play_trajectory(self, init_state, init_info, init_discount=1) -> Tuple[float, float]:
        reward, discounted_reward = super().play_trajectory(init_state, init_info, init_discount)
        self._log_trajectory_end(reward)

        return reward, discounted_reward

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

    def study_agent(self, episode_n, seed=None, card_picker_seed=None):
        root = "../models"
        agent_name = type(self.agent).__name__
        agent_name = f"{root}/{agent_name}"
        env_name = type(self.env).__name__
        epoch_n = self.define_epoch_count_and_save_rate(episode_n)

        os.makedirs(agent_name, exist_ok=True)

        for epoch in range(epoch_n):
            path = f"{agent_name}/model_{epoch}_{env_name}.pt"
            super().study_agent(self.save_rate, seed=seed, card_picker_seed=card_picker_seed)
            self.save_model(path, agent_name, env_name, epoch, is_after_study=False)
        path = f"{agent_name}/model_{epoch_n}_{env_name}.pt"
        self.save_model(path, agent_name, env_name, epoch_n, is_after_study=True)

        return self

    def define_epoch_count_and_save_rate(self, episode_n) -> int:
        if self.save_rate is None:
            epoch_n = 1
            self.save_rate = episode_n
        else:
            epoch_n = math.ceil(episode_n / self.save_rate)

        return epoch_n

    def save_model(self, path, agent_name, env_name, epoch, is_after_study):
        memory = self.agent.memory
        if not self.save_memory:
            self.agent.memory = []
        save_dqn_agent(self.agent, path=path)
        self.agent.memory = memory
        for key in self._logs.keys():
            self.save_log(agent_name, env_name, epoch, key, is_after_study)

    def save_log(self, agent_name, env_name, epoch, key, is_after_study):
        entry = self._logs[key]
        state = entry.get_log_state(is_after_study)
        if state == KeyLogState.DO_NOT_LOG:
            return
        data = entry.data
        if state == KeyLogState.ONLY_LEN_LOG:
            data = len(data)
        data = repr(data)
        with open(f"{agent_name}/{key}_{epoch}_{env_name}.txt", mode="w") as f:
            f.write(data)

    def set_log_state(self, key, state, is_after_study):
        if key not in self._logs.keys():
            return
        self._logs[key].set_log_state(state, is_after_study)

