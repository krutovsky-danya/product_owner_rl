from game.userstory_card.userstory_card_info import UserStoryCardInfo


class TechDebtInfo(UserStoryCardInfo):
    def __init__(self):
        super().__init__(label_val="TechDebt")
        self.hours_debuff_increment = 1
        self.full_hours_debuff = 1
