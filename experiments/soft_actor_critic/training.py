import datetime
import sys

sys.path.append("..")
sys.path.append("../..")

from algorithms.soft_actor_critic import SoftActorCritic
from algorithms.SoftActorCriticFactory import SoftActorCriticFactory
from environment import ProductOwnerEnv
from environment.environments_factory import EnvironmentFactory
from pipeline import LoggingStudy
from training_utils import save_evaluation, save_study_data, make_evaluations


def train(agents_factory, env_factory):
    episode_n = 1500
    trajectory_max_len = 1000

    env: ProductOwnerEnv = env_factory()
    agent: SoftActorCritic = agents_factory(env.state_dim, env.action_n)
    study = LoggingStudy(env, agent, trajectory_max_len)

    study.study_agent(episode_n)

    agent_name = agent.__class__.__name__
    env_name = env.__class__.__name__
    experiment_name = agent_name + "_on_" + env_name

    save_study_data(study, experiment_name)

    evaluations = make_evaluations(study, 1000)

    now = datetime.datetime.now()
    save_evaluation(agent_name, evaluations, now, experiment_name)


if __name__ == "__main__":
    environment_factory = EnvironmentFactory()
    agents_factory = SoftActorCriticFactory()
    for i in range(5):
        for agent_factory in agents_factory.get_agents():
            for env_factory in environment_factory.get_environments():
                train(agent_factory, env_factory)
