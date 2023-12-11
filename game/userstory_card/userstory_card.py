from game.userstory_card.userstory_card_info import UserStoryCardInfo


class UserStoryCard:
    def __init__(self, info: UserStoryCardInfo):
        self.is_movable = True
        self.info = info

    def set_card_info(self, card_info: UserStoryCardInfo):
        self.info = card_info
