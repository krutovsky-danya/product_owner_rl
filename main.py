from pipeline import study_agent
from pipeline.study_agent import study_dqn

from algorithms import deep_q_networks
from algorithms.deep_q_networks import DQN

if __name__ == '__main__':
    agent = DQN(1, 1)