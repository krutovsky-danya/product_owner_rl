import torch
from algorithms.deep_q_networks import DQN


def save_dqn_agent(agent: DQN, path):
    torch.save(agent, path)


def load_dqn_agent(path):
    agent: DQN = torch.load(path)
    agent.eval()
    return agent
