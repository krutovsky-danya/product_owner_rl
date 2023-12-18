import matplotlib.pyplot as plt

from pipeline.study_agent import LoggingStudy
from algorithms.deep_q_networks import DoubleDQN
from environment.environment import ProductOwnerEnv

if __name__ == "__main__":
    env = ProductOwnerEnv()
    state_dim = env.state_dim
    action_n = env.action_n

    agent = DoubleDQN(state_dim, action_n, tau=0.001, epsilon_decrease=1e-6)

    study = LoggingStudy(env, agent, trajecory_max_len=1_000, save_rate=100)

    study.study_agent(1_000)

    rewards = study.rewards_log
    estimates = study.q_value_log

    plt.plot(rewards)
    plt.plot(estimates)
    plt.show()
