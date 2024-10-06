from game.userstory_card.userstory_card_info import UserStoryCardInfo


class TechDebtInfo(UserStoryCardInfo):
    def __init__(self, spawn_sprint, color_storage, random_gen):
        super().__init__("TechDebt", spawn_sprint, color_storage, random_gen)
        self.hours_debuff_increment = 1
        self.full_hours_debuff = 1
