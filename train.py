import datetime

import pandas as pd

from algorithms.agents_factory import DqnAgentsFactory
from environment.environments_factory import EnvironmentFactory
from experiments.show_utils import show_win_rate
from experiments.training_utils import make_evaluations, save_evaluation
from pipeline import LoggingStudy
from pipeline.study_agent import save_dqn_agent


def make_credit_study(trajectory_max_len, episode_n):
    env = EnvironmentFactory().create_credit_env()
    agent = DqnAgentsFactory().create_ddqn(env.state_dim, env.action_n)
    study = LoggingStudy(env, agent, trajectory_max_len)
    study.study_agent(episode_n)
    return study


def main():
    episode_n = 1501
    trajectory_max_len = 1000
    study = make_credit_study(trajectory_max_len, episode_n)
    save_dqn_agent(study.agent, "models/credit_start_model.pt")

    evaluations = make_evaluations(study, 1000)
    now = datetime.datetime.now()
    save_evaluation("credit_start", evaluations, now, "main")

    evaluations = pd.read_csv("evaluations_credit_start.csv")
    show_win_rate(evaluations)


if __name__ == "__main__":
    main()
