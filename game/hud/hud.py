from game import game_global as Global
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from game.backlog_card.card_info import CardInfo


class HUD:
    def __init__(self):
        self.release_available = False

    def increase_progress(self, cards_to_update):
        for i in cards_to_update:
            card: CardInfo = i
            us: UserStoryCardInfo = Global.current_stories[card.us_id]
            us.completed_part += card.base_hours / us.time_to_complete
            if abs(1 - us.completed_part) < 1e-3 or us.completed_part > 1.0:
                self.release_available = True

# increase_progress:
    # должно быть так, чтобы не было бага со временем выполнения
    # us.completed_part += card.base_hours / us.time_to_complete
    # (us.completed_part += card.base_hours / us.time_to_complete * part_of_sprint),
    # если как в godot
