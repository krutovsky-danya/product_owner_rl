import matplotlib.pyplot as plt

from pipeline.study_agent import LoggingStudy
from algorithms.deep_q_networks import DQN, DoubleDQN
from environment.environment import ProductOwnerEnv

if __name__ == "__main__":
    env = ProductOwnerEnv()
    state_dim = env.state_dim
    action_n = env.action_n

    agent = DQN(state_dim, action_n, epsilon_decrease=1e-6)

    study = LoggingStudy(env, agent, trajecory_max_len=1_000, save_rate=100)

    try:
        study.study_agent(300)
    except KeyboardInterrupt:
        pass

    rewards = study.rewards_log
    estimates = study.q_value_log

    plt.plot(rewards)
    plt.plot(estimates)
    plt.xlabel("Trajectory")
    plt.ylabel('Reward')
    plt.show()

    plt.plot(study.sprints_log)
    plt.title('Sprints count')
    plt.xlabel("Trajectory")
    plt.ylabel("Sprint")
    plt.show()
