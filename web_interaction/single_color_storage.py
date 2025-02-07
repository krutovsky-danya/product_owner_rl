from game.game_colors import ColorStorage


class SingleColorStorage(ColorStorage):
    def __init__(self, color):
        super().__init__()
        self.color = color

    def get_unused_color(self, uc_type, random):
        return self.color

    def release_color(self, us_type, color):
        pass
