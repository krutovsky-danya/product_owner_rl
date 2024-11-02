from game.backlog_card.card_info import CardInfo


class BacklogCardImageInfo:
    def __init__(self, color, hours, position):
        self.color = color
        self.hours = hours
        self.position = position

    def _equals_backlog_card(self, value: CardInfo):
        if self.color != value.color:
            return False
        return self.hours == value.base_hours

    def __eq__(self, value: object) -> bool:
        if isinstance(value, CardInfo):
            return self._equals_backlog_card(value)
        if not isinstance(value, BacklogCardImageInfo):
            return False
        if self.color != value.color:
            return False
        if self.hours != value.hours:
            return False
        if self.position != value.position:
            return False
        return True
