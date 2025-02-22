import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

sys.path.append("..")
sys.path.append("../..")

from show_utils import get_wins_stat

def main():
    episodes_n = '1501'
    evals_df = pd.read_csv(f"evaluations_{episodes_n}.csv")
    evals_df['Lose'] = 1 - evals_df['Win']

    evals_groups = evals_df.groupby(["Flag"])

    grouped_wins = evals_groups[["Win", 'Lose']].sum()
    grouped_wins['%'] = grouped_wins['Win'] / (grouped_wins['Win'] + grouped_wins['Lose'])
    print(grouped_wins.to_markdown())

    exp_enabled = evals_df['Flag']
    experiment_wins = evals_df[exp_enabled]['Win'].to_numpy()
    default_wins = evals_df[~exp_enabled]['Win'].to_numpy()
    res = get_wins_stat(default_wins, experiment_wins)
    print(res)

    succes_sprints_df = evals_df[evals_df['Win']]
    grouped_succes_sprints = succes_sprints_df.groupby(['Flag', 'DateTime'])
    grouped_sprints = grouped_succes_sprints['Sprint'].min()
    print(grouped_sprints)

    rewards_df = pd.read_csv(f"train_rewards_{episodes_n}.csv")
    reward_groups = rewards_df.groupby(["Flag", "Trajectory"])["Reward"]
    mean_rewards = reward_groups.mean().reset_index()
    new_reward = mean_rewards[mean_rewards["Flag"]]
    default_rewards = mean_rewards[~mean_rewards["Flag"]]

    plt.plot(new_reward["Trajectory"], new_reward["Reward"], '.', label="Potential")
    plt.plot(default_rewards["Trajectory"], default_rewards["Reward"], '.', label="Default")
    plt.legend()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.xticks(np.arange(0, 1501, 500))
    plt.savefig(f"potential_rewards_{episodes_n}.png")
    plt.show()


if __name__ == "__main__":
    main()
