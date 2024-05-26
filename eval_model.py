import numpy as np
import matplotlib.pyplot as plt

from environment import CreditPayerEnv, TutorialSolverEnv, ProductOwnerEnv
from pipeline.study_agent import load_dqn_agent
from pipeline.base_study import MAX_INNER_SPRINT_ACTION_COUNT
from pipeline import TUTORIAL, CREDIT_FULL, CREDIT_START, CREDIT_END, END
from pipeline.aggregator_study import update_reward_system_config
from environment.reward_sytem import (EmpiricalCreditStageRewardSystem,
                                      EmpiricalRewardSystem,
                                      FullPotentialCreditRewardSystem)

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title


def eval_model(environments, agents, order, repeat_count: int, is_silent: bool):
    rewards, sprints, loyalties, customers, money, wins = [], [], [], [], [], []

    for _ in range(repeat_count):
        reward = play_eval_trajectory(environments, agents, order, is_silent)
        rewards.append(reward)
        update_logs(environments[order[-1]], sprints, loyalties, customers, money, wins)

    customers = np.array(customers)
    loyalties = np.array(loyalties)

    return (np.median(rewards),
            {"money": money,
             "sprints": sprints,
             "loyalty": loyalties,
             "customers": customers,
             "potential money": customers * loyalties * 300,
             "wins": wins})


def play_eval_trajectory(environments, agents, order, is_silent):
    for stage in order:
        assert stage in environments
        assert stage in agents or stage == END
    env = environments[order[-1]]
    full_reward = 0
    env.reset()

    for stage in order:
        if stage == END and not (stage in agents):
            full_reward += play_end(env, is_silent)
        else:
            full_reward += play_some_stage(env,
                                           environments[stage],
                                           agents[stage],
                                           f"{stage} reward",
                                           is_silent)
    if not is_silent:
        print(f"full reward: {full_reward},"
              f"current sprint: {env.game.context.current_sprint}")

    return full_reward


def update_logs(env, sprints, loyalties, customers, money, wins):
    context = env.game.context
    loyalties.append(context.get_loyalty())
    customers.append(context.customers)
    wins.append(int(context.is_victory))
    sprints.append(context.current_sprint)
    money.append(context.get_money())


def play_end(main_env, is_silent=True):
    done = main_env.game.context.get_money() < 0 or main_env.game.context.customers <= 0
    total_reward = 0
    while not done:
        state, reward, done, info = main_env.step(0)
        total_reward += reward

    return total_reward


def play_some_stage(main_env: ProductOwnerEnv, translator_env: ProductOwnerEnv, agent,
                    state_line, is_silent=True):
    translator_env.game = main_env.game
    lost_customers = not main_env.game.context.is_new_game and main_env.game.context.customers <= 0
    done = main_env.game.context.get_money() < 0 or lost_customers
    state = translator_env.recalculate_state()
    info = translator_env.get_info()
    done = done or len(info["actions"]) == 0
    inner_sprint_action_count = 0
    total_reward = 0

    while not done:
        action, inner_sprint_action_count = choose_action(agent, state, info,
                                                          inner_sprint_action_count)
        state, reward, done, info = translator_env.step(action)

        total_reward += reward

    if not is_silent:
        print(f"{state_line}: {total_reward}")

    return total_reward


def choose_action(agent, state, info, inner_sprint_action_count, is_silent=True):
    action = agent.get_action(state, info)
    if action == 0:
        inner_sprint_action_count = 0
    else:
        inner_sprint_action_count += 1
    if inner_sprint_action_count == MAX_INNER_SPRINT_ACTION_COUNT:
        action = 0
        inner_sprint_action_count = 0
        if not is_silent:
            print("enforced next sprint")
    return action, inner_sprint_action_count


def load_agents(paths):
    agents = {}
    for stage, path in paths.items():
        agent = load_dqn_agent(path)
        agent.epsilon = 0
        agents[stage] = agent
    return agents


def get_backlog_environments():
    return {
        TUTORIAL: None,
        CREDIT_FULL: None,
        CREDIT_END: None,
        CREDIT_START: None,
        END: None
    }


def get_reward_systems():
    return {
        TUTORIAL: EmpiricalRewardSystem(config={}),
        CREDIT_FULL: FullPotentialCreditRewardSystem(config={}),
        CREDIT_START: EmpiricalCreditStageRewardSystem(with_late_purchase_punishment=False,
                                                       config={}),
        CREDIT_END: EmpiricalCreditStageRewardSystem(with_late_purchase_punishment=True, config={}),
        END: EmpiricalRewardSystem(config={})
    }


def get_environments(backlog_environments, reward_systems, with_info):
    res = {
        TUTORIAL: TutorialSolverEnv(userstory_env=None,
                                    backlog_env=backlog_environments[TUTORIAL],
                                    with_info=with_info,
                                    reward_system=reward_systems[TUTORIAL]),
        CREDIT_FULL: CreditPayerEnv(userstory_env=None,
                                    backlog_env=backlog_environments[CREDIT_FULL],
                                    with_end=True,
                                    with_info=with_info,
                                    reward_system=reward_systems[CREDIT_FULL]),
        CREDIT_START: CreditPayerEnv(userstory_env=None,
                                     backlog_env=backlog_environments[CREDIT_START],
                                     with_end=False,
                                     with_info=with_info,
                                     reward_system=reward_systems[CREDIT_START]),
        CREDIT_END: CreditPayerEnv(userstory_env=None,
                                   backlog_env=backlog_environments[CREDIT_END],
                                   with_end=True,
                                   with_info=with_info,
                                   reward_system=reward_systems[CREDIT_END]),
        END: ProductOwnerEnv(userstory_env=None,
                             backlog_env=backlog_environments[END],
                             with_info=with_info,
                             reward_system=reward_systems[END])
    }

    for stage, env in res.items():
        update_reward_system_config(env, env.reward_system)
    return res


def show_usual_plots(results):
    for name, result in results.items():
        plt.plot(result, '.')
        plt.xlabel("Trajectory")
        plt.ylabel(name)
        plt.show()


def eval_wins_and_losses(results):
    wins = np.array(results["wins"])
    wins_check = (wins == 1)
    print(f"wins: {len(wins[wins_check])}")
    money = np.array(results["money"])
    print(f"losses: {len(money[money < 0])}")

    return wins_check


def show_plots_with_wins(results, show_plots=True):
    wins_check = eval_wins_and_losses(results)

    if not show_plots:
        return

    trajectories = np.arange(len(wins_check), dtype=np.int32)
    for name in ["money", "sprints"]:
        plt.plot(trajectories[wins_check], np.array(results[name])[wins_check], '.',
                 label="win", color="red")
        plt.plot(trajectories[~wins_check], np.array(results[name])[~wins_check], '.',
                 label="other", color="blue")
        plt.xlabel("Trajectory")
        plt.ylabel(name)
        plt.legend()
        plt.show()


def load_and_eval_model(paths, repeats=10, with_plots=True, is_silent=False, with_info=True):
    agents = load_agents(paths)
    environments = get_environments(get_backlog_environments(), get_reward_systems(), with_info)
    full_order = [TUTORIAL, CREDIT_FULL, END]

    results = eval_model(environments, agents, full_order, repeats, is_silent=is_silent)
    print(results[0])
    results = results[1]

    if with_plots:
        show_usual_plots(results)

    show_plots_with_wins(results, with_plots)


def main():
    tutorial_path = "./DoubleDQN/model_0_TutorialSolverEnv.pt"

    for i in range(1, 50):
        paths = {
            TUTORIAL: tutorial_path,
            CREDIT_FULL: f"./DoubleDQN/model_{i}_CreditPayerEnv.pt"
        }
        print(f"current model: {i}")
        load_and_eval_model(paths, repeats=1000, with_plots=False, is_silent=True, with_info=True)


if __name__ == "__main__":
    main()
