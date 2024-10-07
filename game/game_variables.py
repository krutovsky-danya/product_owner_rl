from game.game_colors import ColorStorage
from game.game_constants import GlobalConstants
from game.common_methods import clamp
from typing import Dict
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo


def save_to_leaderboard(current_sprint):
    # отличается от годота
    print(f"score: {1000000 - current_sprint}")


class GlobalContext:
    def __init__(self) -> None:
        self.current_sprint = 1
        self.current_stories: Dict[int, UserStoryCardInfo] = {}
        self._money = 200000
        self.done = False
        self.current_sprint_hours = 0
        self.current_tech_debt: Dict[int, TechDebtInfo] = {}
        self.available_stories: Dict[int, UserStoryCardInfo] = {}
        self.is_new_game = True
        self.current_bugs: Dict[int, BugUserStoryInfo] = {}
        self._loyalty = 0
        self.customers = 0
        self.blank_sprint_counter = 0
        self.credit = 300000
        self.available_developers_count = 2
        self.worker_cost = 10000
        self.is_first_bug = True
        self.is_first_tech_debt = True
        self.current_room_multiplier = 1
        self.current_rooms_counter = 1
        self.color_storage = ColorStorage()
        self.is_victory = False
        self.is_loss = False

    def get_money(self):
        return self._money

    def set_money(self, money):
        self._money = money
        self.check_money()

    def check_money(self):
        if self._money < 0:
            self.game_over(False)
        elif self._money >= GlobalConstants.MONEY_GOAL:
            self.game_over(True)

    def game_over(self, is_win):
        self.done = True
        if is_win:
            print("win")
            save_to_leaderboard(self.current_sprint)
            self.is_victory = True
        else:
            # print("lose")
            self.is_loss = True

    def get_loyalty(self):
        return self._loyalty

    def set_loyalty(self, value):
        self._loyalty = clamp(value, 0.8, 5)

    def buy_robot(self):
        self._money -= GlobalConstants.NEW_WORKER_COST
        self.available_developers_count += 1
        self.check_money()

    def buy_room(self):
        self._money -= GlobalConstants.NEW_ROOM_COST * self.current_room_multiplier
        self.current_room_multiplier *= GlobalConstants.NEW_ROOM_MULTIPLIER
        self.current_rooms_counter += 1
        self.available_developers_count += 1
        self.check_money()

    def has_enough_money(self, need_money):
        return self._money >= need_money
