from environment import ProductOwnerEnv
from environment.backlog_env import BacklogEnv
from environment.reward_sytem import EmpiricalRewardSystem, FullPotentialCreditRewardSystem, EmpiricalEndStageRewardSystem
from environment.userstory_env import UserstoryEnv
from main import create_usual_agent
from pipeline import AggregatorStudy, STUDY, END, TUTORIAL, CREDIT_START, CREDIT_END
from pipeline.study_agent import load_dqn_agent, save_dqn_agent
from pipeline.aggregator_study import update_reward_system_config
import visualizer


def make_final_sprints_study(agents,
                             order,
                             trajectory_max_len,
                             episode_n,
                             stage,
                             with_info,
                             save_rate=None):
    reward_system = EmpiricalRewardSystem(config={})
    userstory_env = UserstoryEnv(6, 2, 2)
    backlog_env = BacklogEnv(12, 4, 2, 0, 0, 0)
    env = ProductOwnerEnv(userstory_env, backlog_env, with_info=with_info, reward_system=reward_system)
    update_reward_system_config(env, reward_system)
    agent = create_usual_agent(env, trajectory_max_len, episode_n)

    environments = {STUDY: env}
    agents[STUDY] = agent
    study = AggregatorStudy(environments, agents, order, trajectory_max_len, save_rate=save_rate)
    study.study_agent(episode_n)

    agents[stage] = agent
    order.append(stage)

    return study


def main():
    tutorial_model_path = 'models/tutorial_model.pt'
    tutorial_agent = load_dqn_agent(tutorial_model_path)

    credit_start_path = 'models/credit_start_model.pt'
    credit_start_agent = load_dqn_agent(credit_start_path)

    credit_end_path = 'models/credit_end_model.pt'
    credit_end_agent = load_dqn_agent(credit_end_path)

    agents = {
        TUTORIAL: tutorial_agent,
        CREDIT_START: credit_start_agent,
        CREDIT_END: credit_end_agent
    }
    order = [TUTORIAL, CREDIT_START, CREDIT_END]

    study = make_final_sprints_study(agents, order, 1000, 1000, END, True)
    agent = study.agent

    visualizer.show_rewards(study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(study, filename='figures/sprints.png')
    visualizer.show_loss(study, filename='figures/loss.png')

    agent.memory = []
    save_dqn_agent(agent, 'models/final_sprints_model.pt')


if __name__ == '__main__':
    main()
