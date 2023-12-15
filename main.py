import matplotlib.pyplot as plt

from pipeline.study_agent import LoggingStudy
from algorithms.deep_q_networks import DQN
from environment.environment import ProductOwnerEnv

if __name__ == "__main__":
    env = ProductOwnerEnv()
    state_dim = env.state_dim
    action_n = env.action_n

    agent = DQN(state_dim, action_n)

    study = LoggingStudy(env, agent, 1_000, 1_000)

    study.study_agent(1_000)

    rewards = study.rewards_log
    estimates = study.q_value_log

    plt.plot(rewards)
    plt.plot(estimates)
    plt.show()
