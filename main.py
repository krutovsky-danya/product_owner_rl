import os
import matplotlib.pyplot as plt

from pipeline.study_agent import LoggingStudy, load_dqn_agent
from algorithms.deep_q_networks import DQN, DoubleDQN
from environment.environment import ProductOwnerEnv

if __name__ == "__main__":
    env = ProductOwnerEnv()
    state_dim = env.state_dim
    action_n = env.action_n

    agent = DoubleDQN(state_dim, action_n, tau=0.001, epsilon_decrease=1e-5)

    study = LoggingStudy(env, agent, trajecory_max_len=200, save_rate=100)

    try:
        study.study_agent(1000)
    except KeyboardInterrupt:
        pass

    rewards = study.rewards_log
    estimates = study.q_value_log

    os.makedirs('figures', exist_ok=True)

    plt.plot(rewards, '.')
    plt.plot(estimates)
    plt.xlabel("Trajectory")
    plt.ylabel('Reward')
    plt.savefig('figures/rewards.png')
    plt.show()

    plt.plot(study.sprints_log, '.')
    plt.title('Sprints count')
    plt.xlabel("Trajectory")
    plt.ylabel("Sprint")
    plt.savefig('figures/sprints.png')
    plt.show()
