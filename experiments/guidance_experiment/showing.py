import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def get_experiment_files(filenames: List[str], guidance: bool):
    experiment_files = []
    prefix = f"guidance_{guidance}"
    for filename in filenames:
        if os.path.basename(filename).startswith(prefix):
            experiment_files.append(filename)
    return experiment_files


def get_all_data_files(directory_path: str):
    content = os.listdir(directory_path)

    data_files = []
    for filename in content:
        if filename.startswith("guidance"):
            data_files.append(os.path.join(directory_path, filename))
    return data_files


def read_files_data(filenames: List[str]):
    result = []
    for filename in filenames:
        with open(filename, "r") as file:
            data = eval(file.read())
            result.append(data)
    return result


def get_experements_wins(evaluation_filenames: List[str], guidance: bool):
    exp_evaluation_files = get_experiment_files(evaluation_filenames, guidance)
    guidance_evaluation = read_files_data(exp_evaluation_files)
    guidance_evaluation = np.array(guidance_evaluation)
    wins = guidance_evaluation[:, :, 1].sum(axis=1)
    return wins


def main():
    current_dir = os.getcwd()
    files_dir = os.path.join(current_dir, 'episodes_1000')

    data_files = get_all_data_files(files_dir)

    reward_files = []
    evaluation_files = []
    for data_file in data_files:
        if "rewards" in data_file:
            reward_files.append(data_file)
        if "evals" in data_file:
            evaluation_files.append(data_file)

    guidance_rewards_files = get_experiment_files(reward_files, True)
    guidance_rewards = read_files_data(guidance_rewards_files)
    guidance_rewards = np.array(guidance_rewards)

    default_rewards_files = get_experiment_files(reward_files, False)
    default_rewards = read_files_data(default_rewards_files)

    plt.plot(np.mean(guidance_rewards, axis=0), ".", label="Guidance")
    plt.plot(np.mean(default_rewards, axis=0), ".", label="Default")
    plt.legend()
    plt.title('Rewards')
    plt.xlabel('Trajectory')
    plt.ylabel('Rewards')
    plt.grid()
    plt.savefig('guidance_rewards.png')
    plt.show()

    default_wins = get_experements_wins(evaluation_files, False)
    print(default_wins)

    guidance_wins = get_experements_wins(evaluation_files, True)
    print(guidance_wins)


if __name__ == "__main__":
    main()
