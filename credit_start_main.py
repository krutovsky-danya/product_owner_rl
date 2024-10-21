from environment import CreditPayerEnv
from environment.backlog_env import BacklogEnv
from environment.reward_sytem import FullPotentialCreditRewardSystem
from environment.reward_sytem import EmpiricalCreditStageRewardSystem
from pipeline import AggregatorStudy, STUDY, CREDIT_END, CREDIT_FULL, CREDIT_START, TUTORIAL
from environment.userstory_env import UserstoryEnv
from pipeline.study_agent import load_dqn_agent, save_dqn_agent
from pipeline.aggregator_study import update_reward_system_config, KeyLogState
from main import create_usual_agent

import visualizer


def parse_state_from_stage(stage):
    with_end = stage != CREDIT_START
    with_late_purchases_penalty = stage == CREDIT_END
    return with_end, with_late_purchases_penalty


def get_reward_system(stage, with_late_purchases_penalty):
    if stage != CREDIT_FULL:
        return EmpiricalCreditStageRewardSystem(with_late_purchases_penalty, config={})
    else:
        return FullPotentialCreditRewardSystem(config={})


def make_credit_study(agents,
                      order,
                      trajectory_max_len,
                      episode_n,
                      stage,
                      with_info,
                      save_rate=None):
    with_end, with_late_purchases_penalty = parse_state_from_stage(stage)
    reward_system = get_reward_system(stage, with_late_purchases_penalty)
    userstory_env = UserstoryEnv(2, 0, 0)
    backlog_env = BacklogEnv(6, 0, 0, 0, 0, 0)
    env = CreditPayerEnv(userstory_env, backlog_env, with_end=with_end, with_info=with_info,
                         reward_system=reward_system,
                         seed=None, card_picker_seed=None)
    update_reward_system_config(env, reward_system)

    agent = create_usual_agent(env, trajectory_max_len, episode_n)
    agents[STUDY] = agent
    environments = {STUDY: env}

    study = AggregatorStudy(environments, agents, order, trajectory_max_len, save_rate=save_rate,
                            base_epoch_log_state=KeyLogState.DO_NOT_LOG)
    study.set_log_state(study.LOSS_LOG_KEY, KeyLogState.ONLY_LEN_LOG, is_after_study=False)
    study.study_agent(episode_n)

    order.append(stage)
    agents[stage] = agent

    return study


def main():
    tutorial_model_path = 'models/tutorial_model.pt'
    tutorial_agent = load_dqn_agent(tutorial_model_path)
    order = [TUTORIAL]
    agents = {TUTORIAL: tutorial_agent}

    study = make_credit_study(agents, order, 100, 800, CREDIT_START, with_info=True)
    agent = study.agent

    visualizer.show_rewards(study, show_estimates=True, filename='figures/rewards.png')
    visualizer.show_sprints(study, filename='figures/sprints.png')
    visualizer.show_loss(study, filename='figures/loss.png')

    agent.memory = []
    save_dqn_agent(agent, 'models/credit_start_model.pt')

    # end_study = make_credit_study(agents, order, 100, 1400, CREDIT_END, with_info=True)
    # end_agent = end_study.agent

    # visualizer.show_rewards(end_study, show_estimates=True, filename='figures/rewards.png')
    # visualizer.show_sprints(end_study, filename='figures/sprints.png')
    # visualizer.show_loss(end_study, filename='figures/loss.png')

    # end_agent.memory = []
    # save_dqn_agent(end_agent, 'models/credit_end_model.pt')
    

if __name__ == '__main__':
    main()
