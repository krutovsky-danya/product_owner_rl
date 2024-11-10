import sys

sys.path.append("..")

import pandas as pd

from show_utils import show_rewards_fitting, show_win_rate


def main():
    sub_name = f"1500"
    experiments_names = ["DQN", "HardTargetDQN", "SoftTargetDQN", "DoubleDQN"]

    data_postions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"data_{experiment_name}_{sub_name}.csv")
        data_postions.append(data_frame)

    data = pd.concat(data_postions)

    show_rewards_fitting(data, ["Reward"])
    for experiment_name in experiments_names:
        experiment_data = data[data["ExperimentName"] == experiment_name]
        show_rewards_fitting(experiment_data, ["Estimate", "DiscountedReward"])

    evaluation_data_portions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"evaluations_{experiment_name}_{sub_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)


if __name__ == "__main__":
    main()
