from game.backlog_card.card_info import CardInfo
from game import game_global as Global
import random


class Card:
    def __init__(self):
        self.info = CardInfo(hours_val=random.randint(1, 10),
                             color_val=Global.UserCardColor.BLUE,
                             us_id_val=1,
                             label_val="S",
                             card_type_val=Global.UCType.S)
        self.is_movable = True

    def add_data(self, card_info: CardInfo):
        self.info = card_info
