import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_rewards_fitting(sub_name):
    rewards_df = pd.read_csv(f"train_rewards_{sub_name}.csv")
    reward_groups = rewards_df.groupby(["Trajectory", "ExperimentName"])["Reward"]
    mean_rewards = reward_groups.mean().reset_index()

    for exp_name in set(rewards_df["ExperimentName"]):
        rewards = mean_rewards[mean_rewards["ExperimentName"] == exp_name]
        plt.plot(rewards["Trajectory"], rewards["Reward"], ".", label=exp_name)
    plt.legend()
    plt.grid()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.savefig(f"rewards_{sub_name}.png")
    plt.show()


def show_win_rate(sub_name):
    evals_df = pd.read_csv(f"evaluations_{sub_name}.csv")
    wins = evals_df.groupby(["DateTime", "ExperimentName"])["Win"]
    groups_wins = wins.sum()

    print(groups_wins)

    total_wins = groups_wins.reset_index()
    total_wins = total_wins.groupby(["ExperimentName"])["Win"].sum()
    print(total_wins)

    show_win_sprints(evals_df)


def show_win_sprints(evals_df: pd.DataFrame):
    wins = evals_df[evals_df["Win"]]
    print(wins)


def main():
    episode_n = 250
    trajectory_n = 20
    sub_name = f"CreditPayerEnv_{episode_n}_episodes_{trajectory_n}_trajectory_nn_2x256_inner_layer_gamma_0.9"
    show_rewards_fitting(sub_name)
    show_win_rate(sub_name)


if __name__ == "__main__":
    main()
