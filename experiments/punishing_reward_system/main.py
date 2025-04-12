import sys

import matplotlib.pyplot as plt
import pandas as pd

from typing import List

sys.path.append("..")
sys.path.append("../..")

from algorithms.agents_factory import DqnAgentsFactory
from environment.reward_sytem.PunishingRewardSystem import PunishingRewardSystem
from environment.environments_factory import EnvironmentFactory
from pipeline import LoggingStudy


def make_credit_study(trajectory_max_len, episode_n) -> LoggingStudy:
    reward_system = PunishingRewardSystem(config={})
    env = EnvironmentFactory().create_full_env()
    env.with_info = False
    env.reward_system = reward_system

    agent = DqnAgentsFactory().create_ddqn(env.state_dim, env.action_n)

    study = LoggingStudy(env, agent, trajectory_max_len)
    study.study_agent(episode_n)

    return study


def main():
    episode_n = 100
    study = make_credit_study(200, episode_n)
    pd.DataFrame(study.sprints_log).to_csv("sprints.csv")

    plt.plot(study.sprints_log, ".", label="Sprints")
    plt.xlabel("Episode")
    plt.ylabel("Sprint")
    plt.grid()
    plt.legend()
    plt.savefig("punishing_reward_system.png")
    plt.show()


if __name__ == "__main__":
    main()
