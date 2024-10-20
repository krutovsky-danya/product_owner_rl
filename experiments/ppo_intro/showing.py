import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    episode_n = 250
    trajectory_n = 20
    sub_name = f"CreditPayerEnv_{episode_n}_episodes_{trajectory_n}_trajectory_n_2x256_inner_layer"
    rewards_df = pd.read_csv(f"train_rewards_{sub_name}.csv")
    reward_groups = rewards_df.groupby(["Trajectory", "ExperimentName"])["Reward"]
    mean_rewards = reward_groups.mean().reset_index()

    for exp_name in set(rewards_df['ExperimentName']):
        rewards = mean_rewards[mean_rewards['ExperimentName'] == exp_name]
        plt.plot(
        rewards["Trajectory"], rewards["Reward"], ".", label=exp_name
    )
    plt.legend()
    plt.grid()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.savefig(f"rewards_{sub_name}.png")
    plt.show()


if __name__ == "__main__":
    main()
