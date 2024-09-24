import torch

import gymnasium as gym
import matplotlib.pyplot as plt

from algorithms.proximal_policy_optimization import PPO, PPO_Discrete, PPO_Discrete_2


def study_agent(env: gym.Env, agent: torch.nn.Module, episode_n=50, trajectory_n=20):
    total_rewards = []

    for episode in range(episode_n):

        states, actions, rewards, dones = [], [], [], []

        for _ in range(trajectory_n):
            total_reward = 0

            state, _ = env.reset()
            for t in range(200):
                states.append(state)

                action = agent.get_action(state)
                actions.append(action)

                state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)

                total_reward += reward

                if done:
                    break

            total_rewards.append(total_reward)

        agent.fit(states, actions, rewards, dones)

    return total_rewards


env: gym.Env = gym.make("LunarLander-v2", continuous=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO_Discrete_2(state_dim, action_dim)

returns_total_reward = study_agent(env, agent)

plt.plot(returns_total_reward, ".")
plt.title("Total Rewards")
plt.grid()
plt.savefig(f"{agent.__class__.__name__}.png")
# plt.show()
