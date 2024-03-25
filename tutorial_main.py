from environment import TutorialSolverEnv
from pipeline import LoggingStudy
from main import create_usual_agent

from pipeline.study_agent import save_dqn_agent

import visualizer


def make_tutorial_study(trajectory_max_len, episode_n, with_info):
    env = TutorialSolverEnv(with_info=with_info)
    agent = create_usual_agent(env, trajectory_max_len, episode_n)
    study = LoggingStudy(env, agent, trajectory_max_len)
    study.SAVE_MEMORY = False

    study.study_agent(episode_n)
    return study


def main():
    study = make_tutorial_study(trajectory_max_len=100, episode_n=40, with_info=True)
    agent = study.agent

    visualizer.show_rewards(study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(study, filename='figures/sprints.png')
    visualizer.show_loss(study, filename='figures/loss.png')

    agent.memory = []
    save_dqn_agent(agent, 'models/tutorial_model.pt')


if __name__ == "__main__":
    main()
