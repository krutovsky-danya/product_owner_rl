from game.game_constants import UserCardType

class CardInfo:
    def __init__(self, hours_val: int, color_val, us_id_val: int,
                 label_val: str, card_type_val: UserCardType):
        self.color = color_val
        self.base_hours = hours_val
        self.hours = hours_val
        self.us_id = us_id_val
        self.label = label_val
        self.card_type = card_type_val
    
    def __repr__(self) -> str:
        return f'Hours: {self.hours}, Color: {self.color}, Label: {self.label}, Type: {self.card_type}'
