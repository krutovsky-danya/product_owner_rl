import logging
import torch

import gymnasium as gym
import matplotlib.pyplot as plt

from algorithms.proximal_policy_optimization import PPO, PPO_Discrete_2, PPO_Discrete_3
from pipeline.study_agent import save_dqn_agent
from pipeline.episodic_study import study_ppo_agent


env: gym.Env = gym.make("LunarLander-v2", continuous=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO_Discrete_2(state_dim, action_dim)

returns_total_reward = study_ppo_agent(env, agent, 100, 20)

agent_name = agent.__class__.__name__

plt.plot(returns_total_reward, ".")
plt.title("Total Rewards")
plt.grid()
plt.savefig(f"{agent_name}.png")
plt.show()

save_dqn_agent(agent, f'{agent_name}.png')

env: gym.Env = gym.make("LunarLander-v2", continuous=False, render_mode="human")

state, _ = env.reset()
for i in range(500):
    action = agent.get_action(state)
    state, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break