from environment import CreditPayerEnv
from environment.backlog_env import BacklogEnv
from environment.reward_sytem import BoundedEmpiricalCreditStageRewardSystem
from pipeline import AggregatorStudy
from pipeline.study_agent import load_dqn_agent, save_dqn_agent
from pipeline.aggregator_study import update_reward_system_config
from main import create_usual_agent

import visualizer


def make_credit_study(prev_agents, trajectory_max_len, episode_n, with_end, with_info, with_late_purchase_penalty):
    reward_system = BoundedEmpiricalCreditStageRewardSystem(with_late_purchase_punishment=with_late_purchase_penalty, config={})
    env = CreditPayerEnv(with_end=with_end, with_info=with_info, reward_system=reward_system)
    update_reward_system_config(env, reward_system)

    agent = create_usual_agent(env, trajectory_max_len, episode_n)

    agents = prev_agents + [agent]
    study = AggregatorStudy(env, agents, trajectory_max_len)

    study.study_agent(episode_n)

    return study


def main():
    tutorial_model_path = 'models/tutorial_model.pt'
    tutorial_agent = load_dqn_agent(tutorial_model_path)

    study = make_credit_study([tutorial_agent], 100, 1400, with_end=False, with_info=True, with_late_purchase_penalty=False)
    agent = study.agent

    visualizer.show_rewards(study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(study, filename='figures/sprints.png')
    visualizer.show_loss(study, filename='figures/loss.png')

    agent.memory = []
    save_dqn_agent(agent, 'models/credit_start_model.pt')

    end_agents = [tutorial_agent, agent]
    end_study = make_credit_study(end_agents, 100, 1400, with_end=True, with_info=True, with_late_purchase_penalty=True)
    end_agent = end_study.agent

    visualizer.show_rewards(end_study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(end_study, filename='figures/sprints.png')
    visualizer.show_loss(end_study, filename='figures/loss.png')

    end_agent.memory = []
    save_dqn_agent(end_agent, 'models/credit_end_model.pt')
    

if __name__ == '__main__':
    main()
