import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    episode_n = 250
    trajectory_n = 20
    sub_name = f"{episode_n}_episodes_{trajectory_n}_trajectory_n"
    rewards_df = pd.read_csv(f"train_rewards_{sub_name}.csv")
    reward_groups = rewards_df.groupby(["Trajectory"])["Reward"]
    mean_rewards = reward_groups.mean().reset_index()
    default_rewards = mean_rewards

    plt.plot(
        default_rewards["Trajectory"], default_rewards["Reward"], ".", label="Default"
    )
    plt.legend()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.xticks(np.arange(0, 5001, 1000))
    plt.savefig(f"rewards_{sub_name}.png")
    plt.show()


if __name__ == "__main__":
    main()
