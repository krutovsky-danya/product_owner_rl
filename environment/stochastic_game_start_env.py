from environment.environment import ProductOwnerEnv
from game.game_generators import (
    ProductOwnerGame,
    get_buggy_game_1,
    get_buggy_game_2,
    get_buggy_game_3,
    get_game_on_sprint_6,
    get_game_on_sprint_21,
    get_game_on_sprint_26,
)


class StochasticGameStartEnv(ProductOwnerEnv):
    def __init__(
        self,
        userstories_env=None,
        backlog_env=None,
    ):
        super().__init__(
            userstories_env,
            backlog_env,
        )

        self.index = 0
        self.generators = [
            ProductOwnerGame,
            get_game_on_sprint_6,
            get_game_on_sprint_21,
            get_game_on_sprint_26,
            get_buggy_game_1,
            get_buggy_game_2,
            get_buggy_game_3,
        ]

    def reset(self):
        self.game = self.generators[self.index]()
        self.index = (self.index + 1) % len(self.generators)
        self.current_state = self._get_state()
        return self.current_state
