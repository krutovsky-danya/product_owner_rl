import sys

sys.path.append("..")

import pandas as pd

from show_utils import (
    show_rewards_fitting,
    show_win_rate,
    show_estimate_reward_comparison,
)


def show_win_sprint_hist(data: pd.DataFrame):
    win_data = data[data["Win"]].drop(columns=["Win"])

    hist = (
        win_data.groupby(["Sprint", "ExperimentName"])
        .size()
        .unstack()
        .plot(kind="bar", stacked=False)
    )
    hist.set_ylabel("Number of wins")
    hist.set_xlabel("Sprint")
    hist.set_title("Number of wins per sprint")
    hist.legend(title="Experiment")
    hist.get_figure().savefig("wins.png")
    hist.get_figure().show()


def main():
    sub_name = f"1500"
    experiments_names = ["DQN", "DoubleDQN"]

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
        data_frame = pd.read_csv(f"evaluations_{experiment_name}_{sub_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)
    show_win_sprint_hist(evaluation_data)


if __name__ == "__main__":
    main()
