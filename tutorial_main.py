from environment import TutorialSolverEnv
from environment.backlog_env import BacklogEnv
from pipeline import LoggingStudy
from main import create_usual_agent

from environment.reward_sytem import BoundedEmpiricalRewardSystem, EmpiricalRewardSystem
from pipeline.study_agent import save_dqn_agent
from pipeline.aggregator_study import update_reward_system_config

import visualizer


def make_tutorial_study(trajectory_max_len, episode_n, with_info):
    backlog_env = BacklogEnv(4, 0, 0, 0, 0, 0)
    reward_system = EmpiricalRewardSystem(config={'remove_sprint_card_actions': []})
    env = TutorialSolverEnv(backlog_env=backlog_env, with_info=with_info, reward_system=reward_system)
    update_reward_system_config(env, reward_system)
    agent = create_usual_agent(env, trajectory_max_len, episode_n)
    study = LoggingStudy(env, agent, trajectory_max_len)
    study.SAVE_MEMORY = False

    study.study_agent(episode_n)
    return study


def main():
    study = make_tutorial_study(trajectory_max_len=100, episode_n=400, with_info=True)
    agent = study.agent

    visualizer.show_rewards(study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(study, filename='figures/sprints.png')
    visualizer.show_loss(study, filename='figures/loss.png')

    agent.memory = []
    save_dqn_agent(agent, 'models/tutorial_model.pt')


if __name__ == "__main__":
    main()
