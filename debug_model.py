import matplotlib.pyplot as plt
from pipeline import LoggingStudy

from pipeline.study_agent import load_dqn_agent
from algorithms.deep_q_networks import DoubleDQN
from environment.environment import LoggingEnv

if __name__ == "__main__":
    env = LoggingEnv()
    state_dim = env.state_dim
    action_n = env.action_n

    agent = load_dqn_agent('DoubleDQN/model_1.pt')

    study = LoggingStudy(env, agent, trajecory_max_len=1_000, save_rate=1)

    study.study_agent(1)
