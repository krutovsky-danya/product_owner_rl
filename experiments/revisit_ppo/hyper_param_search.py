import optuna
import os
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from algorithms.agents_factory import PPOAgentsFactory
from environment.environments_factory import EnvironmentFactory
from experiments.training_utils import eval_ppo_agent
from pipeline.episodic_study import EpisodicPpoStudy


def ppo_objective(trial: optuna.Trial):
    env = EnvironmentFactory().create_full_env()
    agent_factory = PPOAgentsFactory()
    agent = agent_factory.create_ppo_discrete_logits_guided(env.state_dim, env.action_n)

    agent_factory.epoch_n = trial.suggest_int("epoch_n", 1, 5)
    trajectory_n = trial.suggest_int("trajectory_n", 1, 10)

    trajectory_max_len = 1000

    episode_n = 1500 // trajectory_n

    study = EpisodicPpoStudy(
        env=env,
        agent=agent,
        trajectory_max_len=trajectory_max_len,
    )

    study.study_agent(episode_n=episode_n, trajectory_n=trajectory_n)

    win_rate = 0
    for i in range(100):
        reward, is_win, sprint = eval_ppo_agent(study)
        if is_win:
            win_rate += 1

    return win_rate / 1000


def main():
    ppo_study = optuna.create_study(direction="maximize")
    ppo_study.optimize(ppo_objective, n_trials=1)

    trial = ppo_study.best_trial

    print("Reward: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    fig = optuna.visualization.plot_optimization_history(ppo_study)
    fig.write_html("optimization_history.html")
    fig.write_image("optimization_history.png")
    fig.show()

    fig = optuna.visualization.plot_slice(ppo_study)
    fig.write_html("slice.html")
    fig.write_image("slice.png")
    fig.show()


if __name__ == "__main__":
    main()
