import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "full")[: -w + 1] / w


def show_rewards_fitting(df: pd.DataFrame):
    columns = ["Reward"]
    reward_groups = df.groupby(["Trajectory", "ExperimentName"])[columns]
    mean_rewards = reward_groups.mean().reset_index()

    for experiment_name in set(df["ExperimentName"]):
        for column in columns:
            rewards = mean_rewards[mean_rewards["ExperimentName"] == experiment_name]
            plt.plot(
                rewards["Trajectory"],
                moving_average(rewards[column], 5),
                label=experiment_name,
            )
    plt.legend()
    plt.grid()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.savefig("Rewards.png")
    plt.show()


def show_estimate_reward_comparison(df: pd.DataFrame, agent_name: str):
    columns = ["Estimate", "DiscountedReward"]
    data = df[df["ExperimentName"] == agent_name]
    data = data.groupby(["Trajectory"])[columns]
    mean_data = data.mean().reset_index()

    for column in columns:
        plt.plot(
            mean_data["Trajectory"],
            moving_average(mean_data[column], 5),
            label=column,
        )

    plt.legend()
    plt.grid()
    plt.title(agent_name)
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.savefig(f"Esitmates-Rewards-{agent_name}.png")
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
    wins = wins.groupby(["ExperimentName"]).min()
    print(wins.reset_index())
