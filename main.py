import os
import matplotlib.pyplot as plt
from environment import StochasticGameStartEnv

from pipeline.study_agent import LoggingStudy, load_dqn_agent
from algorithms.deep_q_networks import DQN, DoubleDQN
from environment.environment import ProductOwnerEnv, CreditPayerEnv

if __name__ == "__main__":
    env = CreditPayerEnv()
    state_dim = env.state_dim
    action_n = env.action_n

    trajecory_max_len = 100
    episode_n = 400

    epsilon_decrease = 1 / (trajecory_max_len * episode_n)

    agent = DoubleDQN(state_dim, action_n, gamma=0.9, tau=0.001, epsilon_decrease=epsilon_decrease)

    study = LoggingStudy(env, agent, trajecory_max_len=trajecory_max_len, save_rate=100)

    try:
        study.study_agent(episode_n + 100)
    except KeyboardInterrupt:
        pass

    rewards = study.rewards_log
    estimates = study.q_value_log

    os.makedirs('figures', exist_ok=True)

    plt.plot(rewards, '.', label='Rewards')
    plt.plot(estimates, '.', label='Estimates')
    plt.xlabel("Trajectory")
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('figures/rewards.png')
    plt.show()

    plt.plot(study.sprints_log, '.')
    plt.title('Sprints count')
    plt.xlabel("Trajectory")
    plt.ylabel("Sprint")
    plt.savefig('figures/sprints.png')
    plt.show()

    plt.plot(study.loss_log, '.')
    plt.title('Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig('figures/loss.png')
    plt.show()
