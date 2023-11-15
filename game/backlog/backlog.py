from game import game_global as Global
from game.backlog_card.backlog_card import Card
from game.userstory_card.userstory_card_info import UserStoryCardInfo


class Backlog:
    def __init__(self):
        self.backlog = []
        self.sprint = []

    def can_start_sprint(self):
        hours_to_sum = self.calculate_hours_sum()
        return int(hours_to_sum) != 0 or int(Global.customers) != 0

    def generate_cards(self):
        self.backlog.clear()
        self.sprint.clear()
        for i in Global.current_stories.values():
            us: UserStoryCardInfo = i
            if us.is_decomposed:
                for j in us.related_cards:
                    card = Card()
                    self.backlog.append(card)
                    card.add_data(j)

    def on_start_sprint_pressed(self):
        hours = self.calculate_hours_sum()
        if hours > Global.available_developers_count * Global.developer_hours:
            return
        Global.current_sprint_hours = hours
        return self.get_cards()

    def clear_sprint(self):
        self.sprint.clear()

    def calculate_hours_sum(self) -> int:
        need_hours_sum = Global.current_sprint_hours
        for card in self.sprint:
            need_hours_sum += card.info.hours

        return need_hours_sum

    def get_cards(self):
        cards_info = []

        for card in self.sprint:
            cards_info.append(card.info)

        return cards_info
