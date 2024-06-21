from .base_reward_system import BaseRewardSystem


class EmpiricalEndStageRewardSystem(BaseRewardSystem):
    def __init__(self, config: dict, done_reward_bias=0) -> None:
        super().__init__(config)
        self.wrong_action_reward = -10
        self.lose_reward = -100
        self.done_reward_bias = done_reward_bias

    def get_reward(self, state_old, action, state_new, success) -> float:
        if not success:
            return self.wrong_action_reward
        return self.get_done_reward(state_new) + self.done_reward_bias

    def get_done_reward(self, state_new) -> float:
        done = self.get_done(state_new)
        customers = self.get_customers(state_new)
        if not done and customers > 0:
            return 0
        if self.get_money(state_new) >= 1e6:
            return -self.get_sprint(state_new)
        return self.lose_reward
