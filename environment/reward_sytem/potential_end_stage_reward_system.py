from .base_reward_system import BaseRewardSystem


class PotentialEndStageRewardSystem(BaseRewardSystem):
    def __init__(self, config: dict, done_reward_bias=0, gamma=1) -> None:
        super().__init__(config)
        self.wrong_action_reward = -10
        self.lose_reward = -100
        self.win_reward = 0
        self.done_reward_bias = done_reward_bias
        self.money_weight = 1e-3
        # discount factor gamma
        self.gamma = gamma

    def get_reward(self, state_old, action, state_new, success) -> float:
        if not success:
            return self.wrong_action_reward
        action_reward = self.get_action_reward(state_old, state_new)
        done_reward = self.get_done_reward(state_new) + self.done_reward_bias
        return action_reward + done_reward

    def get_action_reward(self, state_old, state_new):
        return self.gamma * self.eval_state(state_new) - self.eval_state(state_old)

    def eval_state(self, state):
        return self.get_money(state) * self.money_weight

    def get_done_reward(self, state_new) -> float:
        done = self.get_done(state_new)
        customers = self.get_customers(state_new)
        if not done and customers > 0:
            return 0
        if self.get_money(state_new) >= 1e6:
            return self.win_reward
        return self.lose_reward
