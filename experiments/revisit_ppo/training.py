import datetime
import os
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from pipeline.episodic_study import EpisodicPpoStudy
from environment.environments_factory import EnvironmentFactory
from algorithms.agents_factory import PPOAgentsFactory
from experiments.training_utils import eval_ppo_agent, save_study_data, save_evaluation


def make_credit_study(
    trajectory_max_len, episode_n, trajectory_n, agent_factory
) -> EpisodicPpoStudy:
    env = EnvironmentFactory().create_credit_env()

    state_dim = env.state_dim
    action_n = env.action_n

    agent = agent_factory(state_dim, action_n)

    study = EpisodicPpoStudy(env, agent, trajectory_max_len)
    study.study_agent(episode_n, trajectory_n)

    return study


def main(agent_class):
    episode_n = 2000
    trajectory_max_len = 1000
    trajectory_n = 5
    study = make_credit_study(trajectory_max_len, episode_n, trajectory_n, agent_class)
    experiment_name = study.agent.__class__.__name__
    save_study_data(study, experiment_name)

    evaluations = []
    for i in range(100):
        evaluation = eval_ppo_agent(study)
        evaluations.append(evaluation)

    now = datetime.datetime.now()
    save_evaluation("PPO", evaluations, now, experiment_name)


if __name__ == "__main__":
    n = 5
    agent_factory = PPOAgentsFactory()
    agent_factory.epoch_n = 5
    agent_factory.inner_layer = 2048
    for i in range(n):
        main(agent_factory.create_ppo_discrete_logits_guided)
        print(f"Iteration {i + 1} completed.")
