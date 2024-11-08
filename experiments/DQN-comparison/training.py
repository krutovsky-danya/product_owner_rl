import datetime
import sys

sys.path.append("..")
sys.path.append("../..")

from algorithms import DQN
from algorithms.agents_factory import DqnAgentsFactory
from environment import ProductOwnerEnv
from environment.environments_factory import EnvironmentFactory
from pipeline import LoggingStudy
from training_utils import eval_agent, save_rewards, save_evaluation


def train(agent_factory, env_factory):
    episode_n = 1500
    trajectory_max_len = 1000

    env: ProductOwnerEnv = env_factory()

    state_dim = env.state_dim
    action_n = env.action_n

    agent: DQN = agent_factory(state_dim, action_n)

    study = LoggingStudy(env, agent, trajectory_max_len)

    study.study_agent(episode_n)

    now = datetime.datetime.now()
    experiment_name = agent.__class__.__name__

    save_rewards(episode_n, study.rewards_log, now, experiment_name)

    evaluations = []
    for i in range(100):
        evaluation = eval_agent(study)
        evaluations.append(evaluation)

    save_evaluation(episode_n, evaluations, now, experiment_name)


if __name__ == "__main__":
    agents_factory = DqnAgentsFactory()
    environment_factory = EnvironmentFactory()
    train(agents_factory.create_dqn_agent, environment_factory.create_credit_env)
