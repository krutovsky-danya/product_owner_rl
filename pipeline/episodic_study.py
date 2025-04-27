import torch
from environment import ProductOwnerEnv
from algorithms.proximal_policy_optimization import PPO_Discrete_Logits_Guided


class EpisodicPpoStudy:
    def __init__(
        self,
        env: ProductOwnerEnv,
        agent: PPO_Discrete_Logits_Guided,
        trajectory_max_len: int,
    ) -> None:
        self.env: ProductOwnerEnv = env
        self.agent: PPO_Discrete_Logits_Guided = agent
        self.trajectory_max_len = trajectory_max_len
        self.rewards_log = []
        self.q_value_log = []
        self.discounted_rewards_log = []

    def play_trajectory(self):
        initial_state = state = self.env.reset()
        info = self.env.get_info()
        states, actions, rewards, dones, infos = [], [], [], [], []
        total_reward, discounted_reward, discount = 0, 0, 1

        for t in range(self.trajectory_max_len):
            states.append(state)
            infos.append(info)

            action = self.agent.get_action(state, info)
            actions.append(action)

            state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            dones.append(done)

            total_reward += reward
            discounted_reward += reward * discount
            discount *= self.agent.gamma

            if done:
                break

        self._log_trajectory(total_reward, discounted_reward, initial_state)
        return states, actions, rewards, dones, infos

    def _log_trajectory(self, total_reward, discounted_reward, state):
        v_value = self.agent.get_value(state, None).cpu().item()
        self.q_value_log.append(v_value)
        self.rewards_log.append(total_reward)
        self.discounted_rewards_log.append(discounted_reward)

    def play_batch_trajectories(self, trajectory_n: int):
        wins = 0
        containers = states, actions, rewards, dones, infos = [], [], [], [], []

        for _ in range(trajectory_n):
            trajectory_data = self.play_trajectory()
            wins += int(self.env.game.context.is_victory)
            for container, elements in zip(containers, trajectory_data):
                container.extend(elements)

        print(f"Wins count: {wins}")
        return states, actions, rewards, dones, infos

    def study_agent(self, episode_n: int, trajectory_n: int):
        for episode in range(episode_n):
            print(f"Started episode {episode + 1}")
            data = self.play_batch_trajectories(trajectory_n)
            self.agent.fit(*data)
        return self.rewards_log


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
