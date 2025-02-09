import datetime
import sys

sys.path.append("..")
sys.path.append("../..")

from algorithms import DQN
from algorithms.agents_factory import DqnAgentsFactory
from environment import ProductOwnerEnv
from environment.environments_factory import EnvironmentFactory
from pipeline import LoggingStudy
from training_utils import save_evaluation, save_study_data, make_evaluations


def train(agent_factory, env_factory):
    episode_n = 1500
    trajectory_max_len = 1000

    env: ProductOwnerEnv = env_factory()
    agent: DQN = agent_factory(env.state_dim, env.action_n)
    study = LoggingStudy(env, agent, trajectory_max_len)

    experiment_name = agent.__class__.__name__

    study.study_agent(episode_n)

    save_study_data(study, experiment_name)

    evaluations = make_evaluations(study, 1000)

    sub_name = experiment_name + f"_{episode_n}"

    now = datetime.datetime.now()
    save_evaluation(sub_name, evaluations, now, experiment_name)


if __name__ == "__main__":
    agents_factory = DqnAgentsFactory()
    environment_factory = EnvironmentFactory()
    train(agents_factory.create_dqn, environment_factory.create_credit_env)
    train(agents_factory.create_hard_target_dqn, environment_factory.create_credit_env)
    train(agents_factory.create_soft_target_dqn, environment_factory.create_credit_env)
    train(agents_factory.create_ddqn, environment_factory.create_credit_env)
