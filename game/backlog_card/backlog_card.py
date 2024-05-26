from game.backlog_card.card_info import CardInfo


class Card:
    def __init__(self):
        self.info = None
        self.is_movable = True
        self.is_in_sprint = False

    def __repr__(self) -> str:
        return repr(self.info)

    def add_data(self, card_info: CardInfo):
        self.info = card_info
