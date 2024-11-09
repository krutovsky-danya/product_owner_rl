import datetime
import sys

sys.path.append("..")
sys.path.append("../..")

from algorithms import DQN
from algorithms.agents_factory import DqnAgentsFactory
from environment import ProductOwnerEnv
from environment.environments_factory import EnvironmentFactory
from pipeline import LoggingStudy
from training_utils import eval_agent, save_evaluation, save_data


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

    data = zip(
        range(episode_n),
        study.rewards_log,
        study.q_value_log,
        study.discounted_rewards_log,
    )

    sub_name = experiment_name + f'_{episode_n}'
    columns = ["Trajectory", "Reward", "Estimate", "DiscountedReward"]
    save_data(sub_name, data, columns, experiment_name)

    evaluations = []
    for i in range(1000):
        evaluation = eval_agent(study)
        evaluations.append(evaluation)

    save_evaluation(sub_name, evaluations, now, experiment_name)


if __name__ == "__main__":
    agents_factory = DqnAgentsFactory()
    environment_factory = EnvironmentFactory()
    train(agents_factory.create_dqn, environment_factory.create_credit_env)
    train(agents_factory.create_hard_target_dqn, environment_factory.create_credit_env)
    train(agents_factory.create_soft_target_dqn, environment_factory.create_credit_env)
    train(agents_factory.create_ddqn, environment_factory.create_credit_env)
