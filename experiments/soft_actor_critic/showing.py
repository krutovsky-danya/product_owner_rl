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


def main(env_class):
    experiments_names = []
    for agent_class in ['SoftActorCritic', 'SACWithLearnedTemperature']:
        experiment_name = agent_class + "_on_" + env_class
        experiments_names.append(experiment_name)

    data_postions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"data_{experiment_name}_{1500}.csv")
        data_postions.append(data_frame)

    data = pd.concat(data_postions)

    show_rewards_fitting(data)
    for agent_name in experiments_names:
        show_estimate_reward_comparison(data, agent_name)


    evaluation_data_portions = []
    for experiment_name in ['SoftActorCritic', 'SACWithLearnedTemperature']:
        data_frame = pd.read_csv(f"evaluations_{experiment_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)
    show_win_sprint_hist(evaluation_data, 30)


if __name__ == "__main__":
    print(__file__)
    main('ProductOwnerEnv')
    input()
    main('CreditPayerEnv')
