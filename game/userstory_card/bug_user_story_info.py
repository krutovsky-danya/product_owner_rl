from game.userstory_card.userstory_card_info import UserStoryCardInfo


class BugUserStoryInfo(UserStoryCardInfo):
    def __init__(self, spawn_sprint, color_storage, random_gen):
        super().__init__("Bug", spawn_sprint, color_storage, random_gen)
        self.loyalty_debuff = -0.05
        self.loyalty_increment = -0.05
        self.customers_debuff = -0.5
        self.customers_increment = -0.5
