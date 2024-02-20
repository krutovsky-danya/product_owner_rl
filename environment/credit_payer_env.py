from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.environment import ProductOwnerEnv


class CreditPayerEnv(ProductOwnerEnv):
    def __init__(self, userstory_env=None, backlog_env=None, with_sprint=True):
        if userstory_env is None:
            userstory_env = UserstoryEnv(6, 0, 0)
        if backlog_env is None:
            backlog_env = BacklogEnv(12, 0, 0, 12, 0, 0)
        super().__init__(userstory_env, backlog_env, with_sprint)

    def step(self, action: int):
        new_state, reward, done, info = super().step(action)
        if self.game.context.current_sprint == 35:
            done = True
            reward += self.game.context.get_money() / 1e5
        return new_state, reward, done, info