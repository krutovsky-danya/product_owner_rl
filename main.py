import matplotlib.pyplot as plt

from pipeline.study_agent import LoggingStudy
from algorithms.deep_q_networks import DQN

if __name__ == "__main__":
    agent = DQN(1, 1)

    study = LoggingStudy('env', agent, 1_000, 1_000)

    study.study_agent(0 * 1_000)

    rewards = study.rewards_log
    estimates = study.q_value_log

    plt.plot(rewards)
    plt.plot(estimates)
    plt.show()
