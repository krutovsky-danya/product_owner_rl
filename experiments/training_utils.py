import datetime
import os
import sys

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from typing import List

sys.path.append("..")

from environment import CreditPayerEnv
from pipeline import LoggingStudy
from pipeline.episodic_study import EpisodicPpoStudy


def play_forward_with_empty_sprints(env: CreditPayerEnv):
    info = env.get_info()
    done = env.get_done(info)
    total_reward = 0
    context = env.game.context
    while not context.done and context.customers > 0:
        state, reward, done, info = env.step(0)
        total_reward += reward
    if context.customers <= 0:
        context.done = True
        context.is_loss = True


def eval_agent(study: LoggingStudy):
    study.agent.eval()
    state = study.env.reset()
    info = study.env.get_info()
    reward, _ = study.play_trajectory(state, info)
    play_forward_with_empty_sprints(study.env)
    game_context = study.env.game.context
    is_win = game_context.is_victory
    sprint = game_context.current_sprint
    return reward, is_win, sprint


def eval_ppo_agent(study: EpisodicPpoStudy):
    study.agent.eval()
    reward, *_ = study.play_trajectory()
    play_forward_with_empty_sprints(study.env)
    game_context = study.env.game.context
    is_win = game_context.is_victory
    sprint = game_context.current_sprint
    return reward, is_win, sprint


def update_data_frame(path: str, df: pd.DataFrame):
    if os.path.exists(path):
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame()

    data: pd.DataFrame = pd.concat([data, df])
    data.to_csv(path, index=False, float_format="%.5f")


def save_rewards(sub_name: str, rewards_log: List[float], now: str, experiment_name):
    df = pd.DataFrame(
        {
            "Trajectory": list(range(len(rewards_log))),
            "Reward": rewards_log,
        }
    )
    df["DateTime"] = now
    df["ExperimentName"] = experiment_name
    rewards_path = f"train_rewards_{sub_name}.csv"
    update_data_frame(rewards_path, df)


def save_evaluation(sub_name: str, evaluations: List, now: str, experiment_name):
    df = pd.DataFrame(evaluations, columns=["Reward", "Win", "Sprint"])
    df["DateTime"] = now
    df["ExperimentName"] = experiment_name
    evaluations_path = f"evaluations_{sub_name}.csv"
    update_data_frame(evaluations_path, df)


def get_wins_stat(a_wins: np.ndarray, b_wins: np.ndarray):
    wins = np.array([a_wins.sum(), b_wins.sum()])
    sizes = np.array([a_wins.size, b_wins.size])
    loses = sizes - wins

    print(wins, loses)
    res = chi2_contingency([wins, loses])
    return res
