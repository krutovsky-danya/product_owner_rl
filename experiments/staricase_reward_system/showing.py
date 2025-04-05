import os
import sys

sys.path.append("../..")

import pandas as pd

from experiments.show_utils import (
    show_win_rate,
    show_win_sprint_hist,
    show_win_rate_statistical_significance,
    show_sprint_statistical_significance,
    show_rewards_fitting,
    show_estimate_reward_comparison,
)


def main():
    experiments_names = ['DoubleDQN_staricase_reward_system']

    reward_data_frames = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"data_{experiment_name}_{1500}.csv")
        reward_data_frames.append(data_frame)

    data = pd.concat(reward_data_frames)

    show_rewards_fitting(data)
    for agent_name in experiments_names:
        show_estimate_reward_comparison(data, agent_name)

    evaluation_data_portions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"evaluations_{experiment_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)
    # show_win_sprint_hist(evaluation_data, 30)


if __name__ == "__main__":
    main()
