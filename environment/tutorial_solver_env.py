from environment.environment import ProductOwnerEnv
from environment.userstory_env import UserstoryEnv
from environment.backlog_env import BacklogEnv


class TutorialSolverEnv(ProductOwnerEnv):
    def __init__(self, userstory_env=None, backlog_env=None, with_info=True, reward_system=None,
                 seed=None, card_picker_seed=None):
        if userstory_env is None:
            userstory_env = UserstoryEnv(2, 0, 0)
        if backlog_env is None:
            backlog_env = BacklogEnv(4, 0, 0, 0, 0, 0)
        super().__init__(userstory_env, backlog_env, with_info, reward_system,
                         seed=seed, card_picker_seed=card_picker_seed)

    def get_done(self, info):
        done_game = super().get_done(info)
        done_tutorial = not self.game.context.is_new_game
        return done_game or done_tutorial
