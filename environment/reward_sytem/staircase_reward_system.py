from .base_reward_system import BaseRewardSystem

from environment.Action import Action


class StaircaseRewardSystem(BaseRewardSystem):
    def __init__(self, sprint_edge=100) -> None:
        super().__init__({})
        self.money_boundary = 1e6
        self.sprint_edge = sprint_edge
        self.important_actions = [
            Action.START_SPRINT,
            Action.RELEASE,
        ]

    def get_done(self, state) -> bool:
        is_new_game = self.is_new_game(state)
        customers = self.get_customers(state)
        have_lost_customers = customers < 0 and not is_new_game
        return state["done"] or have_lost_customers

    def get_reward(self, state_old, action, state_new, success) -> float:
        if not success:
            return -1

        if self.get_done(state_new):
            return self.get_end_game_reward(state_new)

        if action in self.important_actions:
            return 0.01

        return 0.001

    def get_end_game_reward(self, state) -> float:
        sprint = self.get_sprint(state)
        win = self.get_money(state) >= self.money_boundary

        valuable_sprint = min(sprint, self.sprint_edge)

        sprint_difference = self.sprint_edge - valuable_sprint

        if win:
            return abs(sprint_difference)

        return -abs(sprint_difference)
