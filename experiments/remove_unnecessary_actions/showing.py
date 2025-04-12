import os
import sys

sys.path.append("../..")

import pandas as pd

from experiments.show_utils import (
    show_win_rate,
    show_win_sprint_hist,
    show_rewards_fitting,
    show_estimate_reward_comparison,
)


def main():
    experiments_names = ["remove_unnecessary_actions", "DoubleDQN_256"]

    data_postions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"data_{experiment_name}_{1500}.csv")
        data_postions.append(data_frame)

    data = pd.concat(data_postions)

    show_rewards_fitting(data)
    for agent_name in experiments_names:
        show_estimate_reward_comparison(data, agent_name)

    evaluation_data_portions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"evaluations_{experiment_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)
    show_win_sprint_hist(evaluation_data, 30)


if __name__ == "__main__":
    if not os.path.exists("data_DoubleDQN_256_1500.csv"):
        df = pd.read_csv("../neural_net_size/data_DoubleDQN_256_1500.csv")
        df.to_csv("data_DoubleDQN_256_1500.csv", index=False)
    if not os.path.exists("evaluations_DoubleDQN_256.csv"):
        df = pd.read_csv("../neural_net_size/evaluations_DoubleDQN_256.csv")
        df.to_csv("evaluations_DoubleDQN_256.csv", index=False)
    main()
