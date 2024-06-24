import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from typing import List


def get_experiment_files(filenames: List[str], flag: bool):
    experiment_files = []
    prefix = f"bound_{flag}"
    for filename in filenames:
        if os.path.basename(filename).startswith(prefix):
            experiment_files.append(filename)
    return experiment_files


def get_all_data_files(directory_path: str):
    content = os.listdir(directory_path)

    data_files = []
    for filename in content:
        data_files.append(os.path.join(directory_path, filename))
    return data_files


def read_files_data(filenames: List[str]):
    result = []
    for filename in filenames:
        with open(filename, "r") as file:
            data = eval(file.read())
            result.append(data)
    return result


def get_experements_wins(evaluation_filenames: List[str], modified: bool):
    exp_evaluation_files = get_experiment_files(evaluation_filenames, modified)
    evaluation = read_files_data(exp_evaluation_files)
    evaluation = np.array(evaluation)
    wins = evaluation[:, :, 1]
    return wins


def get_wins_stat(a_wins: np.ndarray, b_wins: np.ndarray):
    wins = np.array([a_wins.sum(), b_wins.sum()])
    sizes = np.array([a_wins.size, b_wins.size])
    loses = sizes - wins

    print(wins, loses)
    res = chi2_contingency([wins, loses])
    return res


def main():
    current_dir = os.getcwd()
    files_dir = os.path.join(current_dir, "episodes_1501")

    data_files = get_all_data_files(files_dir)

    reward_files = []
    evaluation_files = []
    for data_file in data_files:
        if "rewards" in data_file:
            reward_files.append(data_file)
        if "evals" in data_file:
            evaluation_files.append(data_file)

    modified_rewards_files = get_experiment_files(reward_files, True)
    modified_rewards = read_files_data(modified_rewards_files)

    default_rewards_files = get_experiment_files(reward_files, False)
    default_rewards = read_files_data(default_rewards_files)

    plt.plot(np.mean(modified_rewards, axis=0), ".", label="bounding")
    plt.plot(np.mean(default_rewards, axis=0) / 100, ".", label="default")
    plt.legend()
    plt.title("Rewards")
    plt.xlabel("trajectory")
    plt.ylabel("rewards")
    plt.savefig("bounding_rewards.png")
    plt.show()

    default_wins = get_experements_wins(evaluation_files, False)
    print(default_wins.sum(axis=1))

    exp_wins = get_experements_wins(evaluation_files, True)
    print(exp_wins.sum(axis=1))

    res = get_wins_stat(default_wins, exp_wins)
    print(res)


if __name__ == "__main__":
    main()
