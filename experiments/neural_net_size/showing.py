import os
import sys

sys.path.append("../..")

import pandas as pd

from experiments.show_utils import (
    show_win_rate,
    show_win_sprint_hist,
    show_win_rate_statistical_significance,
    show_sprint_statistical_significance,
)


def main():
    cpu_experiment_names = ['DoubleDQN_256', 'DoubleDQN_512']
    gpu_experiment_names = ['DoubleDQN_256_GPU', 'DoubleDQN_512_GPU', 'DoubleDQN_1024', 'DoubleDQN_2048']
    experiments_names = cpu_experiment_names + gpu_experiment_names


    evaluation_data_portions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"evaluations_{experiment_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)
    show_win_sprint_hist(evaluation_data, 30)

    print('CPU 256 vs 512')
    show_win_rate_statistical_significance(evaluation_data, *cpu_experiment_names)
    show_sprint_statistical_significance(evaluation_data, *cpu_experiment_names)

    print('CPU vs GPU (256)')
    show_win_rate_statistical_significance(evaluation_data, 'DoubleDQN_256', 'DoubleDQN_256_GPU')
    show_sprint_statistical_significance(evaluation_data, 'DoubleDQN_256', 'DoubleDQN_256_GPU')

    print('CPU vs GPU (512)')
    show_win_rate_statistical_significance(evaluation_data, 'DoubleDQN_512', 'DoubleDQN_512_GPU')
    show_sprint_statistical_significance(evaluation_data, 'DoubleDQN_512', 'DoubleDQN_512_GPU')

    print("GPU 256 vs 512")
    show_win_rate_statistical_significance(evaluation_data, 'DoubleDQN_256_GPU', 'DoubleDQN_512_GPU')
    show_sprint_statistical_significance(evaluation_data, 'DoubleDQN_256_GPU', 'DoubleDQN_512_GPU')

    print("GPU 512 vs 2048")
    show_win_rate_statistical_significance(evaluation_data, 'DoubleDQN_512_GPU', 'DoubleDQN_2048')
    show_sprint_statistical_significance(evaluation_data, 'DoubleDQN_512_GPU', 'DoubleDQN_2048')


if __name__ == "__main__":
    print(__file__)
    if not os.path.exists('data_DoubleDQN_512_1500.csv'):
        df = pd.read_csv('../end-game/data_full_game_baseline_1500.csv')
        df['ExperimentName'] = 'DoubleDQN_512'
        df.to_csv('data_DoubleDQN_512_1500.csv', index=False)
    if not os.path.exists('evaluations_DoubleDQN_512.csv'):
        df = pd.read_csv('../end-game/evaluations_full_game_baseline.csv')
        df['ExperimentName'] = 'DoubleDQN_512'
        df.to_csv('evaluations_DoubleDQN_512.csv', index=False)
    main()
