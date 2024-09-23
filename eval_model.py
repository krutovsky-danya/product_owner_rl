import numpy as np
import matplotlib.pyplot as plt

from environment import CreditPayerEnv, TutorialSolverEnv, ProductOwnerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from pipeline.study_agent import load_dqn_agent
from pipeline.base_study import MAX_INNER_SPRINT_ACTION_COUNT
from pipeline import TUTORIAL, CREDIT_FULL, CREDIT_START, CREDIT_END, END
from pipeline.aggregator_study import update_reward_system_config
from environment.reward_sytem import (EmpiricalCreditStageRewardSystem,
                                      EmpiricalRewardSystem,
                                      FullPotentialCreditRewardSystem,
                                      EmpiricalEndStageRewardSystem)

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22
FIGURE_SIZE = (9, 6)
# FIGURE_SIZE = (10, 8)

plt.rc("font", size=MEDIUM_SIZE)         # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # font size of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # font size of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend font size
plt.rc("figure", titlesize=BIGGER_SIZE)  # font size of the figure title

ROOT_DIR = "../models"
COLOR_SCHEMES = [{"win": "red", "other": "blue"},
                 {"win": "orange", "other": "black"}]

MONEY_LABEL = "money"
SPRINTS_LABEL = "sprints"
LOYALTY_LABEL = "loyalty"
CUSTOMERS_LABEL = "customers"
POTENTIAL_MONEY_LABEL = "potential money"
WINS_LABEL = "wins"
REWARDS_LABEL = "rewards"
WIN_CHECK_LABEL = "win check"

FIRST_PLACE_SPRINTS = 47
SECOND_PLACE_SPRINTS = 48
THIRD_PLACE_SPRINTS = 50


def eval_wins_and_losses(wins, money):
    wins = np.array(wins)
    wins_check = (wins == 1)
    print(f"wins: {len(wins[wins_check])}")
    money = np.array(money)
    print(f"losses: {len(money[money < 0])}")

    return wins_check


def eval_model(environments, agents, order, repeat_count: int, is_silent: bool):
    rewards, sprints, loyalties, customers, money, wins = [], [], [], [], [], []

    for _ in range(repeat_count):
        reward = play_eval_trajectory(environments, agents, order, is_silent)
        rewards.append(reward)
        update_logs(environments[order[-1]], sprints, loyalties, customers, money, wins)

    customers = np.array(customers)
    loyalties = np.array(loyalties)

    return (np.median(rewards),
            {MONEY_LABEL: money,
             SPRINTS_LABEL: sprints,
             LOYALTY_LABEL: loyalties,
             CUSTOMERS_LABEL: customers,
             POTENTIAL_MONEY_LABEL: customers * loyalties * 300,
             WINS_LABEL: wins,
             REWARDS_LABEL: rewards,
             WIN_CHECK_LABEL: eval_wins_and_losses(wins, money),
             })


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
        TUTORIAL: BacklogEnv(4, 0, 0, 0, 0, 0),
        CREDIT_FULL: BacklogEnv(12, 4, 2, 0, 0, 0),
        CREDIT_END: BacklogEnv(12, 4, 2, 0, 0, 0),
        CREDIT_START: BacklogEnv(12, 4, 2, 0, 0, 0),
        END: BacklogEnv(12, 4, 2, 0, 0, 0)
    }


def get_userstory_environments():
    return {
        TUTORIAL: None,
        CREDIT_FULL: UserstoryEnv(6, 2, 2),
        CREDIT_END: UserstoryEnv(6, 2, 2),
        CREDIT_START: UserstoryEnv(6, 2, 2),
        END: UserstoryEnv(6, 2, 2)
    }


def get_reward_systems():
    return {
        TUTORIAL: EmpiricalRewardSystem(config={}),
        CREDIT_FULL: FullPotentialCreditRewardSystem(config={}),
        CREDIT_START: EmpiricalCreditStageRewardSystem(with_late_purchase_punishment=False,
                                                       config={}),
        CREDIT_END: EmpiricalCreditStageRewardSystem(with_late_purchase_punishment=True, config={}),
        END: EmpiricalEndStageRewardSystem(config={})
    }


def get_environments(userstory_environments, backlog_environments, reward_systems, with_info):
    res = {
        TUTORIAL: TutorialSolverEnv(userstory_env=userstory_environments[TUTORIAL],
                                    backlog_env=backlog_environments[TUTORIAL],
                                    with_info=with_info,
                                    reward_system=reward_systems[TUTORIAL]),
        CREDIT_FULL: CreditPayerEnv(userstory_env=userstory_environments[CREDIT_FULL],
                                    backlog_env=backlog_environments[CREDIT_FULL],
                                    with_end=True,
                                    with_info=with_info,
                                    reward_system=reward_systems[CREDIT_FULL]),
        CREDIT_START: CreditPayerEnv(userstory_env=userstory_environments[CREDIT_START],
                                     backlog_env=backlog_environments[CREDIT_START],
                                     with_end=False,
                                     with_info=with_info,
                                     reward_system=reward_systems[CREDIT_START]),
        CREDIT_END: CreditPayerEnv(userstory_env=userstory_environments[CREDIT_END],
                                   backlog_env=backlog_environments[CREDIT_END],
                                   with_end=True,
                                   with_info=with_info,
                                   reward_system=reward_systems[CREDIT_END]),
        END: ProductOwnerEnv(userstory_env=userstory_environments[END],
                             backlog_env=backlog_environments[END],
                             with_info=with_info,
                             reward_system=reward_systems[END]),
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


def show_plots_with_wins(results: dict, colors=None, alpha=1.0):
    for name in [MONEY_LABEL, SPRINTS_LABEL]:
        plt.figure(figsize=FIGURE_SIZE)
        for label, result in results.items():
            wins_check = result[WIN_CHECK_LABEL]
            trajectories = np.arange(len(wins_check), dtype=np.int32)
            win_color = "red" if colors is None else colors[label]["win"]
            other_color = "blue" if colors is None else colors[label]["other"]
            plt.plot(trajectories[wins_check], np.array(result[name])[wins_check], '.',
                     label=f"win {label}", color=win_color, alpha=alpha)
            plt.plot(trajectories[~wins_check], np.array(result[name])[~wins_check], '.',
                     label=f"other {label}", color=other_color, alpha=alpha)
        plt.xlabel("Trajectory")
        plt.ylabel(name)
        plt.legend()
        plt.show()


def show_rewards_sprints_plot(results: dict, colors=None, alpha=1.0):
    plt.figure(figsize=FIGURE_SIZE)
    for label, result in results.items():
        x = np.array(result[REWARDS_LABEL])
        y = np.array(result[SPRINTS_LABEL])
        wins_check = result[WIN_CHECK_LABEL]
        win_color = "red" if colors is None else colors[label]["win"]
        other_color = "blue" if colors is None else colors[label]["other"]
        plt.plot(x[wins_check], y[wins_check], '.',
                 label=f"win {label}", color=win_color, alpha=alpha)
        plt.plot(x[~wins_check], y[~wins_check], '.',
                 label=f"other {label}", color=other_color, alpha=alpha)
    plt.xlabel("Rewards")
    plt.ylabel("Sprints")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2)
    plt.show()


def load_and_get_results(paths, repeats=10, is_silent=True, with_info=True):
    agents = load_agents(paths)
    environments = get_environments(get_userstory_environments(), get_backlog_environments(),
                                    get_reward_systems(), with_info)
    full_order = [CREDIT_FULL, END]

    results = eval_model(environments, agents, full_order, repeats, is_silent=is_silent)
    print(results[0])
    return results[1]


def transform_result_to_metrics(result, repeats):
    sprints = np.array(result[SPRINTS_LABEL])
    wins_check = result[WIN_CHECK_LABEL]
    return (len(wins_check[wins_check]),
            (len(sprints[sprints < THIRD_PLACE_SPRINTS]) / repeats,
             len(sprints[sprints < SECOND_PLACE_SPRINTS]) / repeats,
             len(sprints[sprints < FIRST_PLACE_SPRINTS]) / repeats))


NO_PLOTS = 0
WITH_USUAL_PLOTS = 1
WITH_WINS_PLOTS = 2
WITH_REWARDS_SPRINTS_PLOTS = 4


def check_plots_state(state, condition):
    return state & condition


def load_and_eval_models(models,
                         repeats=10, plots_state=NO_PLOTS, is_silent=False, with_info=True,
                         colors=None, alpha=1.0):
    results = {}
    names = []
    metrics = {}

    for label, model in models.items():
        result = load_and_get_results(model, repeats, is_silent, with_info)
        results[label] = result
        names = result.keys()
        metrics[label] = transform_result_to_metrics(result, repeats)

    if colors is None:
        colors = {}
        for i, label in enumerate(models.keys()):
            colors[label] = COLOR_SCHEMES[i % len(COLOR_SCHEMES)]

    if check_plots_state(plots_state, WITH_USUAL_PLOTS):
        show_usual_plots(results, names, colors, alpha)
    if check_plots_state(plots_state, WITH_WINS_PLOTS):
        show_plots_with_wins(results, colors, alpha)
    if check_plots_state(plots_state, WITH_REWARDS_SPRINTS_PLOTS):
        show_rewards_sprints_plot(results, colors, alpha)

    return metrics


def main():
    repeats = 10
    win_coefficient = 0.5
    tutorial_path = f"{ROOT_DIR}/DoubleDQN/model_0_TutorialSolverEnv.pt"
    credit_path = f"{ROOT_DIR}/DoubleDQN - (67) - with desc/model_197_CreditPayerEnv.pt"
    model_label = ""
    best_models = []
    best_models_with_wins_count = []

    for i in [215, 387]:
        paths = {
            model_label: {
                # TUTORIAL: tutorial_path,
                CREDIT_FULL: credit_path,
                END: f"{ROOT_DIR}/DoubleDQN - (67--+) - with desc/model_{i}_ProductOwnerEnv.pt"
            }
        }
        print(f"current model: {i}")
        wins_count, sprints_res = load_and_eval_models(paths,
                                                       repeats=repeats,
                                                       plots_state=WITH_WINS_PLOTS,
                                                       is_silent=True,
                                                       with_info=True)[model_label]
        if wins_count > win_coefficient * repeats:
            best_models.append(i)
            best_models_with_wins_count.append((i, wins_count / repeats, *sprints_res))
        print(f"done model: {i}")
    print(best_models)
    print(best_models_with_wins_count)


def compare_between_models(with_info=True, repeats=10, is_silent=True):
    models = {
        # "обычный": {
        #     CREDIT_FULL: f"{ROOT_DIR}/DoubleDQN - (60++--) - with desc/model_174_CreditPayerEnv.pt"
        # },
        # "батч-нормализация": {
        #     CREDIT_FULL: f"{ROOT_DIR}/DoubleDQN - (50+) - with desc/model_174_CreditPayerEnv.pt"
        # }
        "обычный": {
            CREDIT_FULL: f"{ROOT_DIR}/DoubleDQN - (67) - with desc/model_197_CreditPayerEnv.pt"
        },
        "батч-нормализация": {
            CREDIT_FULL: f"{ROOT_DIR}/DoubleDQN - (67) - with desc/model_197_CreditPayerEnv.pt"
        }
    }

    load_and_eval_models(models,
                         repeats=repeats,
                         plots_state=WITH_USUAL_PLOTS | WITH_REWARDS_SPRINTS_PLOTS,
                         is_silent=is_silent,
                         with_info=with_info,
                         alpha=0.5)


if __name__ == "__main__":
    main()
