import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List
from scipy.stats import chi2_contingency, mannwhitneyu


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "full")[: -w + 1] / w


def get_wins_stat(a_wins: np.ndarray, b_wins: np.ndarray):
    wins = np.array([a_wins.sum(), b_wins.sum()])
    sizes = np.array([a_wins.size, b_wins.size])
    loses = sizes - wins
    res = chi2_contingency([wins, loses])
    return res


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

    total_wins = data.groupby(["ExperimentName"]).agg({"Win": ["sum", "count"]})
    total_wins["WinRate"] = total_wins["Win"]["sum"] / total_wins["Win"]["count"]
    print(total_wins)

    show_win_sprints(data)


def show_win_sprints(evals_df: pd.DataFrame):
    wins = evals_df[evals_df["Win"]]
    wins = wins.groupby(["ExperimentName"]).min()
    print(wins.reset_index())


def show_win_sprint_hist(data: pd.DataFrame, sprints_count=None):
    win_data = data[data["Win"]].drop(columns=["Win"])

    hist = (
        win_data.groupby(["Sprint", "ExperimentName"])
        .size()
        .unstack()
        .plot(kind="bar", stacked=False)
    )
    hist.set_ylabel("Number of wins")
    hist.set_xlabel("Sprint")
    hist.set_xlim(-1, sprints_count)
    hist.set_title("Number of wins per sprint")
    hist.legend(title="Experiment")
    hist.get_figure().savefig("wins.png")
    hist.get_figure().show()


def show_win_rate_statistical_significance(
    data: pd.DataFrame, experiment_name_a: str, experiment_name_b: str
):
    a_wins = data[data["ExperimentName"] == experiment_name_a]["Win"].values
    b_wins = data[data["ExperimentName"] == experiment_name_b]["Win"].values
    res = get_wins_stat(a_wins, b_wins)
    print("Win rate significance p-value:", res.pvalue)
    print("Win rate significance statistic:", res.statistic)


def show_sprint_statistical_significance(
    data: pd.DataFrame, experiment_name_a: str, experiment_name_b: str
):
    win_data = data[data["Win"]].drop(columns=["Win"])
    a_data = win_data[win_data["ExperimentName"] == experiment_name_a]
    b_data = win_data[win_data["ExperimentName"] == experiment_name_b]

    a_win_sprints = a_data["Sprint"].values
    b_win_sprints = b_data["Sprint"].values

    mannwhitneyu_res = mannwhitneyu(a_win_sprints, b_win_sprints, alternative="two-sided")

    print("Win sprint in A group is equal B group p-value:", mannwhitneyu_res.pvalue)
    print("Win sprint in A group is equal B group statistic:", mannwhitneyu_res.statistic)

    mannwhitneyu_res = mannwhitneyu(a_win_sprints, b_win_sprints, alternative="less")

    print("Sprint in A group is less than in b group p-value:", mannwhitneyu_res.pvalue)
    print(
        "Sprint in A group is less than in b group statistic:",
        mannwhitneyu_res.statistic,
    )
