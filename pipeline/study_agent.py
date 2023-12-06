import torch

from algorithms.deep_q_networks import DQN


def study_dqn(env, agent: DQN, episode_n, trajectory_max_len=1000000, silent=False):
    rewards_log = []
    q_value_log = []

    for episode in range(episode_n):
        total_reward = 0

        state = env.reset()

        with torch.no_grad():
            q_values = agent.q_function(torch.tensor(state))
            q_value_log.append(q_values.max())
        for t in range(trajectory_max_len):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            agent.fit(state, action, reward, done, next_state)

            state = next_state

            if done:
                break

        rewards_log.append(total_reward)
        if not silent:
            print(f"episode: {episode}, total_reward: {total_reward}")

    return rewards_log, q_value_log
