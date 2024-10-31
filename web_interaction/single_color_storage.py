from game.game_colors import ColorStorage

class SingleColorStorage(ColorStorage):
    def __init__(self, color):
        self.color = color
    
    def get_unused_color(self, uc_type, random):
        return self.color