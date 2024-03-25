from environment import CreditPayerEnv, TutorialSolverEnv, ProductOwnerEnv
import numpy as np
from pipeline.study_agent import load_dqn_agent, save_dqn_agent
import matplotlib.pyplot as plt
from pipeline.base_study import MAX_INNER_SPRINT_ACTION_COUNT

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def eval_some_model(env: ProductOwnerEnv, agents, backlog_envs, repeat_count: int, is_silent: bool):
    rewards, sprints, loyalties, customers, money, wins = [], [], [], [], [], []

    for _ in range(repeat_count):
        reward = eval_agents_trajectory(env, agents, backlog_envs, is_silent)
        rewards.append(reward)
        update_logs(env, sprints, loyalties, customers, money, wins)

    customers = np.array(customers)
    loyalties = np.array(loyalties)

    return (np.median(rewards),
            {"money": money,
             "sprints": sprints,
             "loyalty": loyalties,
             "customers": customers,
             "potential money": customers * loyalties * 300,
             "wins": wins})


def eval_agents_trajectory(env: ProductOwnerEnv, agents, backlog_envs, is_silent):
    stage = len(agents)
    if stage == 2:
        assert isinstance(env, CreditPayerEnv)
    full_reward = 0

    if stage > 0:
        full_reward += play_tutorial(env, agents[0], backlog_envs[0], is_silent)
    if stage > 1:
        with_end = False if stage > 2 else env.with_end
        full_reward += play_credit_payment(env, agents[1], backlog_envs[1],
                                           is_silent, with_end=with_end)
    if stage > 2:
        full_reward += play_credit_payment(env, agents[2], backlog_envs[2],
                                           is_silent, with_end=True)
    if stage > 3:
        full_reward += play_some_stage(env, env, agents[3], "end", is_silent)
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


def play_tutorial(main_env, tutorial_agent, backlog_env, is_silent=True):
    main_env.reset()
    env = TutorialSolverEnv(backlog_env=backlog_env, with_info=main_env.with_info)
    return play_some_stage(main_env, env, tutorial_agent, "tutorial reward", is_silent)


def play_credit_payment(main_env, credit_agent, backlog_env, is_silent=True, with_end=False):
    current_sprint = main_env.game.context.current_sprint
    state_line = "credit reward" if current_sprint < 7 else "credit end reward"
    env = CreditPayerEnv(backlog_env=backlog_env, with_end=with_end, with_info=main_env.with_info)
    return play_some_stage(main_env, env, credit_agent, state_line, is_silent)


def play_end(main_env, end_agent, is_silent=True):
    return play_some_stage(main_env, main_env, end_agent, "end reward", is_silent)


def play_some_stage(main_env: ProductOwnerEnv, translator_env: ProductOwnerEnv, agent,
                    state_line, is_silent=True):
    translator_env.IS_SILENT = is_silent
    translator_env.game = main_env.game
    done = main_env.game.context.get_money() < 0
    state = translator_env._get_state()
    info = translator_env.get_info()
    inner_sprint_action_count = 0
    total_reward = 0

    while not done:
        action, inner_sprint_action_count = choose_action(agent, state, info,
                                                          inner_sprint_action_count)
        state, reward, done, info = translator_env.step(action)

        total_reward += reward

    print(f"{state_line}: {total_reward}")

    return total_reward


def choose_action(agent, state, info, inner_sprint_action_count, is_silent=True):
    action = agent.get_action(state, info)
    if action == 0:
        inner_sprint_action_count = 0
    else:
        inner_sprint_action_count += 1
    if inner_sprint_action_count > MAX_INNER_SPRINT_ACTION_COUNT:
        action = 0
        inner_sprint_action_count = 0
        if not is_silent:
            print("enforced next sprint")
    return action, inner_sprint_action_count


def load_agents():
    agent_tutorial = load_dqn_agent("./models/current/tutorial_agent.pt")
    agent_tutorial.epsilon = 0
    agent_credit = load_dqn_agent("./models/current/credit_start_agent.pt")
    agent_credit.epsilon = 0
    agent_credit_end = load_dqn_agent("./models/current/credit_end_agent.pt")
    agent_credit_end.epsilon = 0
    agent_end = load_dqn_agent("./models/current/end_agent.pt")
    return [agent_tutorial, agent_credit, agent_credit_end, agent_end]


def define_backlog_environments():
    return [None] * 4


def eval_model():
    backlog_environments = define_backlog_environments()
    env = ProductOwnerEnv(backlog_env=backlog_environments[-1], with_info=True)
    env.IS_SILENT = True

    results = eval_some_model(env, load_agents(), backlog_environments, 10, is_silent=True)
    print(results[0])
    results = results[1]

    for name, result in results.items():
        plt.plot(result, '.')
        plt.xlabel("Trajectory")
        plt.ylabel(name)
        plt.show()

    wins = np.array(results["wins"])
    wins_check = (wins == 1)
    print(f"wins: {len(wins[wins_check])}")
    money = np.array(results["money"])
    print(f"losses: {len(money[money < 0])}")

    trajectories = np.arange(len(wins), dtype=np.int32)

    for name in ["money", "sprints"]:
        plt.plot(trajectories[wins_check], np.array(results[name])[wins_check], '.',
                 label="win", color="red")
        plt.plot(trajectories[~wins_check], np.array(results[name])[~wins_check], '.',
                 label="other", color="blue")
        plt.xlabel("Trajectory")
        plt.ylabel(name)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    eval_model()
