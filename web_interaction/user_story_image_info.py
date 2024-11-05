from game.userstory_card.userstory_card import UserStoryCard


class UserStoryImageInfo:
    def __init__(self, color, loyalty, customers, position) -> None:
        self.color = color
        self.loyalty = loyalty
        self.customers = customers
        self.position = position

    def _equals_to_game_user_story(self, user_story: UserStoryCard):
        card_info = user_story.info
        if abs(self.loyalty - card_info.loyalty) > 1e-4:
            return False
        if abs(self.customers - card_info.customers_to_bring) > 1e-4:
            return False
        return True

    def __eq__(self, value: object) -> bool:
        if isinstance(value, UserStoryCard):
            return self._equals_to_game_user_story(value)
        if not isinstance(value, UserStoryImageInfo):
            return False
        return (
            self.color == value.color
            and self.loyalty == value.loyalty
            and self.customers == value.customers
            and self.position == value.position
        )

    def __repr__(self) -> str:
        return f"UserStoryImageInfo({self.color}, {self.loyalty}, {self.customers}, {self.position})"
