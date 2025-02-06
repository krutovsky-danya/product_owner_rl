from game.game_constants import GlobalConstants
from game.game_variables import GlobalContext
from game.backlog_card.backlog_card import Card
from game.userstory_card.userstory_card_info import UserStoryCardInfo

from typing import List


class Backlog:
    def __init__(self, context: GlobalContext):
        self.context = context
        self.backlog: List[Card] = []
        self.sprint: List[Card] = []

    def get_max_hours(self) -> int:
        return self.context.available_developers_count * GlobalConstants.developer_hours

    def can_start_sprint(self):
        hours_to_sum = self.calculate_hours_sum()
        if hours_to_sum > self.get_max_hours():
            return False
        return hours_to_sum != 0 or abs(self.context.customers) > 0

    def generate_cards(self):
        self.backlog.clear()
        self.sprint.clear()
        for i in self.context.current_stories.values():
            us: UserStoryCardInfo = i
            if us.is_decomposed:
                for j in us.related_cards:
                    card = Card()
                    self.backlog.append(card)
                    card.add_data(j)

    def on_start_sprint_pressed(self):
        hours = self.calculate_hours_sum()
        if hours > self.get_max_hours():
            return
        self.context.current_sprint_hours = hours
        return self.get_cards()

    def clear_sprint(self):
        self.sprint.clear()

    def calculate_hours_sum(self) -> int:
        need_hours_sum = self.context.current_sprint_hours
        for card in self.sprint:
            need_hours_sum += card.info.hours

        return need_hours_sum

    def get_available_hours(self) -> int:
        return self.get_max_hours() - self.calculate_hours_sum()

    def get_cards(self):
        cards_info = []

        for card in self.sprint:
            cards_info.append(card.info)

        return cards_info
