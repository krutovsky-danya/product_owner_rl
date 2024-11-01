class BacklogCardImageInfo:
    def __init__(self, color, hours, position):
        self.color = color
        self.hours = hours
        self.position = position

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BacklogCardImageInfo):
            return False
        if self.color != value.color:
            return False
        if self.hours != value.hours:
            return False
        if self.position != value.position:
            return False
        return True
