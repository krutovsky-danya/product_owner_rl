from environment.backlog_env import BacklogEnv
from environment.environment import ProductOwnerEnv


class CreditPayerEnv(ProductOwnerEnv):
    def __init__(self, common_userstories_count=4, backlog_env=None):
        if backlog_env is None:
            backlog_env = BacklogEnv(12, 0, 0, 12, 0, 0)
        super().__init__(common_userstories_count, 0, 0, backlog_env)

    def step(self, action: int):
        new_state, reward, done, info = super().step(action)
        if self.game.context.current_sprint == 35:
            done = True
            reward += self.game.context.get_money() / 1e5
        return new_state, reward, done, info