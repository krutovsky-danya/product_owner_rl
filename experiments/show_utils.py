import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List


def show_rewards_fitting(df: pd.DataFrame, columns: List):
    reward_groups = df.groupby(["Trajectory", "ExperimentName"])[columns]
    mean_rewards = reward_groups.mean().reset_index()

    for experiment_name in set(df["ExperimentName"]):
        for column in columns:
            rewards = mean_rewards[mean_rewards["ExperimentName"] == experiment_name]
            plt.plot(rewards["Trajectory"], rewards[column], ".", label=experiment_name + '_' + column)
    plt.legend()
    plt.grid()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.savefig("_".join(columns) + ".png")
    plt.show()


def show_win_rate(data: pd.DataFrame):
    wins = data.groupby(["DateTime", "ExperimentName"])["Win"]
    win_groups = wins.sum()

    print(win_groups)

    total_wins = win_groups.reset_index()
    total_wins = total_wins.groupby(["ExperimentName"])["Win"].sum()
    print(total_wins)

    show_win_sprints(data)


def show_win_sprints(evals_df: pd.DataFrame):
    wins = evals_df[evals_df["Win"]]
    wins = wins.groupby(['ExperimentName']).min()
    print(wins.reset_index())
