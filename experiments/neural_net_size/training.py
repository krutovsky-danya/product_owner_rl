import datetime
import sys
import torch

sys.path.extend(["..", "../.."])

from algorithms import DQN
from algorithms.agents_factory import DqnAgentsFactory
from environment import ProductOwnerEnv
from environment.environments_factory import EnvironmentFactory
from pipeline import LoggingStudy
from training_utils import save_evaluation, save_study_data, make_evaluations


def train(agents_factory: DqnAgentsFactory, env_factory, episode_n=1500, trajectory_max_len=1000):
    """Train a DQN agent in the given environment."""
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    env: ProductOwnerEnv = env_factory()
    agent: DQN = agents_factory.create_ddqn(env.state_dim, env.action_n)
    study = LoggingStudy(env, agent, trajectory_max_len)

    study.study_agent(episode_n)

    experiment_name = f"{agent.__class__.__name__}_{agents_factory.q_function_embeding_size}_{device}"
    print(f"Experiment name: {experiment_name}")

    save_study_data(study, experiment_name)
    evaluations = make_evaluations(study, 1000)

    now = datetime.datetime.now()
    save_evaluation(experiment_name, evaluations, now, experiment_name)


def main():
    """Main function to run training experiments."""
    agents_factory = DqnAgentsFactory()
    environment_factory = EnvironmentFactory()
    embeding_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    num_experiments = 5

    for embeding_size in embeding_sizes:
        agents_factory.q_function_embeding_size = embeding_size
        for i in range(num_experiments):
            print(f"Running experiment {i + 1}/{num_experiments} for embedding size {embeding_size}")
            train(agents_factory, environment_factory.create_full_env)


if __name__ == "__main__":
    main()
