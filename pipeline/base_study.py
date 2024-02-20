from environment import ProductOwnerEnv
from algorithms.deep_q_networks import DQN

MAX_INNER_SPRINT_ACTION_COUNT = 20


class BaseStudy:
    def __init__(self, env: ProductOwnerEnv, agent, trajectory_max_len) -> None:
        self.env = env
        self.agent: DQN = agent
        self.trajectory_max_len = trajectory_max_len

    def fit_agent(self, state, action, reward, done, next_state):
        return self.agent.fit(state, action, reward, done, next_state)

    def play_trajectory(self, init_state):
        total_reward = 0
        state = init_state
        t = 0
        inner_sprint_action_count = 0
        done = False

        while not done:
            action, inner_sprint_action_count = self._choose_action(state,
                                                                    inner_sprint_action_count)
            next_state, reward, done, _ = self.env.step(action)

            self.fit_agent(state, action, reward, done, next_state)

            state = next_state
            total_reward += reward

            t += 1
            done = (t == self.trajectory_max_len)

        return total_reward

    def _choose_action(self, state, inner_sprint_action_count):
        chosen_action = self.agent.get_action(state)
        if chosen_action == 0:
            inner_sprint_action_count = 0
        else:
            inner_sprint_action_count += 1
        if inner_sprint_action_count > MAX_INNER_SPRINT_ACTION_COUNT:
            chosen_action = 0
            inner_sprint_action_count = 0
            if not self.env.IS_SILENT:
                print("enforced next sprint")
        return chosen_action, inner_sprint_action_count

    def study_agent(self, episode_n: int):
        for episode in range(episode_n):
            state = self.env.reset()
            self.play_trajectory(state)
