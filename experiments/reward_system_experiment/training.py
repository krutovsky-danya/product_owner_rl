import datetime
import os
import sys

import pandas as pd

from typing import List

sys.path.append("..")
sys.path.append("../..")

from algorithms import DoubleDQN
from environment import CreditPayerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.reward_sytem import (
    EmpiricalCreditStageRewardSystem,
    FullPotentialCreditRewardSystem,
)
from pipeline.aggregator_study import update_reward_system_config
from pipeline import LoggingStudy
from training_utils import eval_agent, save_rewards, save_evaluation
from main import create_usual_agent


def make_credit_study(trajectory_max_len, episode_n, potential):
    userstory_env = UserstoryEnv(4, 0, 0)
    backlog_env = BacklogEnv(12, 0, 0, 0, 0, 0)
    if potential:
        reward_system = FullPotentialCreditRewardSystem(config={}, coefficient=1)
    else:
        reward_system = EmpiricalCreditStageRewardSystem(True, config={})
    env = CreditPayerEnv(
        userstory_env,
        backlog_env,
        with_end=True,
        with_info=True,
        reward_system=reward_system,
    )
    update_reward_system_config(env, reward_system)

    state_dim = env.state_dim
    action_n = env.action_n

    agent = DoubleDQN(
        state_dim,
        action_n,
        gamma=0.9,
        tau=0.001,
        epsilon_decrease=1e-4,
        batch_size=64,
        lr=1e-3,
        epsilon_min=0.01,
    )

    study = LoggingStudy(env, agent, trajectory_max_len)
    study.study_agent(episode_n)

    return study


def main(potential):
    episode_n = 1501
    study = make_credit_study(200, episode_n, potential)
    now = datetime.datetime.now()
    save_rewards(episode_n, study.rewards_log, now, potential)

    evaluations = []
    for i in range(100):
        evaluation = eval_agent(study)
        evaluations.append(evaluation)

    save_evaluation(episode_n, evaluations, now, potential)


if __name__ == "__main__":
    n = 1
    for i in range(n):
        main(True)
        main(False)
