import sys

sys.path.append("..")

import pandas as pd

from show_utils import (
    show_rewards_fitting,
    show_win_rate,
    show_estimate_reward_comparison,
    show_win_sprint_hist,
    show_win_rate_statistical_significance,
    show_sprint_statistical_significance,
)


def main():
    sub_name = f"1500"
    experiments_names = ["DoubleDQN", "Knapsack_DoubleDQN"]

    data_postions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"data_{experiment_name}_{sub_name}.csv")
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
    show_win_sprint_hist(evaluation_data)
    show_win_rate_statistical_significance(evaluation_data, *experiments_names)
    show_sprint_statistical_significance(evaluation_data, *experiments_names)


if __name__ == "__main__":
    main()
