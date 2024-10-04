import logging
import torch

import matplotlib.pyplot as plt

from algorithms.proximal_policy_optimization import PPO, PPO_Discrete_Softmax, PPO_Discrete_Logits, PPO_Discrete_Softmax_Advantage, PPO_Discrete_Logits_Advantage
from algorithms.proximal_policy_optimization import PPO_Discrete_Softmax_Guided, PPO_Discrete_Logits_Guided
from environment import ProductOwnerEnv, CreditPayerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.reward_sytem.full_potential_credit_reward_system import (
    FullPotentialCreditRewardSystem,
)
from pipeline.study_agent import save_dqn_agent
from pipeline.episodic_study import study_ppo_agent, EpisodicPpoStudy
from pipeline.aggregator_study import update_reward_system_config

userstory_env = UserstoryEnv()
backlog_env = BacklogEnv()
reward_system = FullPotentialCreditRewardSystem(config={}, coefficient=1)
env = CreditPayerEnv(
    userstory_env,
    backlog_env,
    with_end=True,
    with_info=True,
    reward_system=reward_system,
)
update_reward_system_config(env, reward_system)
state_dim = env.state_dim
action_n = env.action_n

agent = PPO_Discrete_Logits_Guided(state_dim, action_n)

study = EpisodicPpoStudy(env, agent, 200)

returns_total_reward = study.study_agent(100, 20)

env_name = env.__class__.__name__
agent_name = agent.__class__.__name__

plt.plot(returns_total_reward, ".")
plt.title(agent_name)
plt.xlabel('Trajectory')
plt.ylabel('Reward')
plt.grid()
plt.savefig(f"{env_name}_{agent_name}.png")
plt.show()

save_dqn_agent(agent, f"{env_name}_{agent_name}.pt")
