import os
import sys

sys.path.append("../..")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from experiments.show_utils import (
    show_win_rate,
    show_win_sprint_hist,
    show_win_rate_statistical_significance,
    show_sprint_statistical_significance,
)


def get_embedings_size(experiment_name: str) -> int:
    """
    Get the embedding size from the experiment name.
    """
    _, embed_size, *_ = experiment_name.split("_")
    return int(embed_size)


def calculate_win_rate(evaluation_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the win rate and embedding size for each experiment.
    """
    win_rate = evaluation_data.groupby(["ExperimentName"]).agg(
        {"Win": ["sum", "count"]}
    )

    win_rate["WinRate"] = win_rate["Win"]["sum"] / win_rate["Win"]["count"]
    win_rate["EmbeddingSize"] = win_rate.index.map(get_embedings_size)

    win_rate = win_rate.sort_values(by="EmbeddingSize")

    return win_rate.reset_index()


def main():
    cpu_experiment_names = []
    gpu_experiment_names = []
    for embedding_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        if embedding_size < 2048:
            cpu_experiment_name = f"DoubleDQN_{embedding_size}_CPU"
        gpu_experiment_name = f"DoubleDQN_{embedding_size}_CUDA"

        cpu_experiment_names.append(cpu_experiment_name)
        gpu_experiment_names.append(gpu_experiment_name)

    experiments_names = cpu_experiment_names + gpu_experiment_names

    evaluation_data_portions = []
    for experiment_name in experiments_names:
        data_frame = pd.read_csv(f"evaluations_{experiment_name}.csv")
        evaluation_data_portions.append(data_frame)

    evaluation_data = pd.concat(evaluation_data_portions)

    show_win_rate(evaluation_data)
    show_win_sprint_hist(evaluation_data, 30)

    win_rate = calculate_win_rate(evaluation_data)

    # Separate GPU and CPU win rates
    gpu_win_rate = win_rate[win_rate["ExperimentName"].isin(gpu_experiment_names)]
    cpu_win_rate = win_rate[win_rate["ExperimentName"].isin(cpu_experiment_names)]

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=gpu_win_rate,
        x="EmbeddingSize",
        y="WinRate",
        marker="o",
        label="GPU",
        color="blue",
    )
    sns.lineplot(
        data=cpu_win_rate,
        x="EmbeddingSize",
        y="WinRate",
        marker="o",
        label="CPU",
        color="orange",
    )

    # Customize the plot
    plt.title("Win Rate by Embedding Size", fontsize=16)
    plt.xlabel("Embedding Size", fontsize=12)
    plt.ylabel("Win Rate", fontsize=12)
    plt.ylim(0, 1)
    plt.xscale("log", base=2)
    plt.yticks(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Device", fontsize=10, title_fontsize=12)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("win_rate_by_embedding_size.png")
    plt.show()


if __name__ == "__main__":
    main()
