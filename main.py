import os
import matplotlib.pyplot as plt
from environment import StochasticGameStartEnv
from environment import CreditPayerEnv
from pipeline import LoggingStudy, ConfidenceStudy

from pipeline.study_agent import load_dqn_agent
from algorithms.deep_q_networks import DQN, DoubleDQN
from environment.environment import ProductOwnerEnv, CreditPayerEnv
import numpy as np


def get_agent_generator(env: ProductOwnerEnv, trajectory_max_len, episode_n):
    state_dim = env.state_dim
    action_n = env.action_n

    trajectory_max_len = 100
    episode_n = 400

    epsilon_decrease = 1 / (trajectory_max_len * episode_n)

    def agent_generator():
        return DoubleDQN(
            state_dim, action_n, gamma=0.9, tau=0.001, epsilon_decrease=epsilon_decrease
        )

    return agent_generator


if __name__ == "__main__":
    env = CreditPayerEnv()

    episode_n = 500
    trajectory_max_len = 100

    agent_generator = get_agent_generator(env, trajectory_max_len, episode_n)

    study = ConfidenceStudy(env, agent_generator, trajectory_max_len)

    try:
        study.study_agents(episode_n, 5)
    except KeyboardInterrupt:
        pass

    rewards = study.rewards_log
    estimates = study.q_value_log

    os.makedirs("figures", exist_ok=True)

    for points, label in [
        (study.rewards_logs, "estimates"),
        (study.q_value_logs, "estimates"),
    ]:
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        plt.plot(mean, label=label)
        xs = np.arange(0, len(mean))
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)

    # plt.plot(rewards, '.', label='Rewards')
    # plt.plot(estimates, '.', label='Estimates')
    plt.xlabel("Trajectory")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("figures/rewards.png")
    plt.show()

    # plt.plot(study.sprints_log, ".")
    # plt.title("Sprints count")
    # plt.xlabel("Trajectory")
    # plt.ylabel("Sprint")
    # plt.savefig("figures/sprints.png")
    # plt.show()

    # plt.plot(study.loss_log, ".")
    # plt.title("Loss")
    # plt.xlabel("Steps")
    # plt.ylabel("Loss")
    # plt.savefig("figures/loss.png")
    # plt.show()
