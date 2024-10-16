import datetime
import os
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from algorithms.proximal_policy_optimization import PPO_Discrete_Logits_Guided_Advantage, PPO_Discrete_Logits_Guided
from environment import CreditPayerEnv, ProductOwnerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.reward_sytem import FullPotentialCreditRewardSystem
from pipeline.aggregator_study import update_reward_system_config
from pipeline import LoggingStudy
from pipeline.episodic_study import EpisodicPpoStudy
from experiments.training_utils import eval_ppo_agent, save_rewards, save_evaluation


def make_credit_study(trajectory_max_len, episode_n, trajectory_n) -> EpisodicPpoStudy:
    userstory_env = UserstoryEnv(4, 0, 0)
    backlog_env = BacklogEnv(12, 0, 0, 0, 0, 0)
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

    study = EpisodicPpoStudy(env, agent, trajectory_max_len)
    study.study_agent(episode_n, trajectory_n)

    return study


def main():
    episode_n = 250
    trajectory_n = 20
    study = make_credit_study(200, episode_n, trajectory_n)
    env_name = study.env.__class__.__name__
    experiment_name = study.agent.__class__.__name__
    data_sub_name = f"{env_name}_{episode_n}_episodes_{trajectory_n}_trajectory_n"
    now = datetime.datetime.now()
    save_rewards(data_sub_name, study.rewards_log, now, experiment_name)

    evaluations = []
    for i in range(100):
        evaluation = eval_ppo_agent(study)
        evaluations.append(evaluation)

    save_evaluation(data_sub_name, evaluations, now, experiment_name)


if __name__ == "__main__":
    n = 3
    for i in range(n):
        main()
