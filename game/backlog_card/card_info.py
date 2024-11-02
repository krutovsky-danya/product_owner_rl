from game.game_constants import UserCardType


class CardInfo:
    def __init__(
        self,
        hours_val: int,
        color_val,
        us_id_val: int,
        label_val: str,
        card_type_val: UserCardType,
    ):
        self.color = color_val
        self.base_hours = hours_val
        self.hours = hours_val
        self.us_id = us_id_val
        self.label = label_val
        self.card_type = card_type_val

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CardInfo):
            return False
        if self.color != value.color:
            return False
        if self.base_hours != value.base_hours:
            return False
        return True

    def __repr__(self) -> str:
        return f'CardInfo({self.hours}, {self.color}, {self.us_id}, {self.label}, {self.card_type})'
