from game.backlog_card.card_info import CardInfo


class Card:
    def __init__(self):
        self.info = None
        self.is_movable = True

    def add_data(self, card_info: CardInfo):
        self.info = card_info
