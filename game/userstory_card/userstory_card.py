from game.userstory_card.userstory_card_info import UserStoryCardInfo


class UserStoryCard:
    def __init__(self, info: UserStoryCardInfo):
        self.is_movable = True
        self.info = info
        self.is_in_release = False

    def set_card_info(self, card_info: UserStoryCardInfo):
        self.info = card_info

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, UserStoryCard):
            return False
        return self.info == value.info

    def __repr__(self) -> str:
        return f"UserStoryCard({repr(self.info)}"
