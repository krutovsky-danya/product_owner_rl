from environment.reward_system.base_reward_system import BaseRewardSystem


class EncouragingRewardSystem(BaseRewardSystem):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def get_reward(self, state_old, action, state_new, success) -> float:
        if not success:
            return -0.01

        if self.get_done(state_new):
            if self.get_money(state_new) == 0:
                return -5.0
            if self.get_customers(state_new) <= 0:
                return -5.0
            return 5.0

        if action == 0:
            return 1.0
        return 0.0
