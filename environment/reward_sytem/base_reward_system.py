from environment.reward_sytem.reward_adapter import RewardAdapter


class BaseRewardSystem(RewardAdapter):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def get_reward(self, state_old, action, state_new, success) -> float:
        return 0
