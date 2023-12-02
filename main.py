import matplotlib.pyplot as plt

from pipeline.study_agent import study_dqn
from algorithms.deep_q_networks import DQN

if __name__ == '__main__':
    agent = DQN(1, 1)

    rewards, estimates = study_dqn(None, agent, 0)

    plt.plot(rewards)
    plt.plot(estimates)
    plt.show()