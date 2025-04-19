from typing import Tuple
from environment import ProductOwnerEnv
from algorithms.deep_q_networks import DQN

MAX_INNER_SPRINT_ACTION_COUNT = 20


class BaseStudy:
    def __init__(self, env: ProductOwnerEnv, agent, trajectory_max_len) -> None:
        self.env = env
        self.agent: DQN = agent
        self.trajectory_max_len = trajectory_max_len

    def fit_agent(self, state, info, action, reward, done, next_state, next_info):
        return self.agent.fit(state, info, action, reward, done, next_state, next_info)

    def play_trajectory(self, init_state, init_info, init_discount=1):
        total_reward = 0
        total_discounted_reward = 0
        discount = init_discount
        state = init_state
        info = init_info
        t = 0
        inner_sprint_action_count = 0
        done = self.env.get_done(info)

        while not done:
            action = self.agent.get_action(state, info)
            action, inner_sprint_action_count = self._choose_action(action,
                                                                    inner_sprint_action_count)
            next_state, reward, done, next_info = self.env.step(action)

            self.fit_agent(state, info, action, reward, done, next_state, next_info)

            state = next_state
            info = next_info
            total_reward += reward
            total_discounted_reward += reward * discount
            discount *= self.agent.gamma

            t += 1
            done = done or (t == self.trajectory_max_len)

        return total_reward, total_discounted_reward

    def _choose_action(self, action, inner_sprint_action_count) -> Tuple[int, int]:
        if action == 0:
            inner_sprint_action_count = 0
        else:
            inner_sprint_action_count += 1
        if inner_sprint_action_count == MAX_INNER_SPRINT_ACTION_COUNT:
            action = 0
            inner_sprint_action_count = 0
        return action, inner_sprint_action_count

    def study_agent(self, episode_n: int, seed=None, card_picker_seed=None):
        for episode in range(episode_n):
            state = self.env.reset(seed=seed, card_picker_seed=card_picker_seed)
            info = self.env.get_info()
            self.play_trajectory(state, info)

    def train(self, mode: bool = True, epsilon: float = 0, epsilon_decrease=None):
        self.agent.train(mode, epsilon)
        if epsilon_decrease is not None:
            self.agent.epsilon_decrease = epsilon_decrease

    def eval(self):
        self.train(mode=False)
