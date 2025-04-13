import torch
from algorithms.deep_q_networks import DQN


def save_dqn_agent(agent: DQN, path):
    agent.memory = None
    torch.save(agent, path)


def load_dqn_agent(path):
    agent: DQN = torch.load(path)
    agent.eval()
    return agent
