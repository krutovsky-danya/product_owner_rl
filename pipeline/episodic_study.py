import torch
from environment import ProductOwnerEnv
from algorithms.proximal_policy_optimization import PPO


class EpisodicPpoStudy:
    def __init__(self, env: ProductOwnerEnv, agent: PPO, trajectory_max_len: int) -> None:
        self.env : ProductOwnerEnv = env
        self.agent: PPO = agent
        self.trajectory_max_len = trajectory_max_len

    def play_trajectory(self):
        total_reward = 0
        state = self.env.reset()
        states, actions, rewards, dones = [], [], [], []
        for t in range(self.trajectory_max_len):
            states.append(state)

            action = self.agent.get_action(state)
            actions.append(action)

            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            dones.append(done)

            total_reward += reward

            if done:
                break
        return total_reward, states, actions, rewards, dones

    def study_agent(self, episode_n: int, trajectory_n: int):
        total_rewards = []
        for episode in range(episode_n):
            print(f'Started episode {episode + 1}')
            states, actions, rewards, dones = [], [], [], []

            for _ in range(trajectory_n):
                tr_total_reward, tr_states, tr_actions, tr_rewards, tr_dones = (
                    self.play_trajectory()
                )
                total_rewards.append(tr_total_reward)
                states.extend(tr_states)
                actions.extend(tr_actions)
                rewards.extend(tr_rewards)
                dones.extend(tr_dones)

            self.agent.fit(states, actions, rewards, dones)
        return total_rewards


def study_ppo_agent(env, agent: torch.nn.Module, episode_n=50, trajectory_n=20):
    total_rewards = []

    for episode in range(episode_n):
        states, actions, rewards, dones = [], [], [], []

        for _ in range(trajectory_n):
            total_reward = 0

            state, _ = env.reset()
            for t in range(200):
                states.append(state)

                action = agent.get_action(state)
                actions.append(action)

                state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)

                total_reward += reward

                if done:
                    break

            total_rewards.append(total_reward)

        agent.fit(states, actions, rewards, dones)

    return total_rewards
