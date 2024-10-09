from enum import Enum
from copy import copy
from random import Random

from game.game_constants import UserCardType


UserCardColor = Enum("UserCardColor",
                     ["BLUE", "GREEN", "ORANGE", "PINK", "PURPLE", "RED", "YELLOW"])


class ColorStorage:
    def __init__(self) -> None:
        self.used_colors = {UserCardType.S: [], UserCardType.M: [], UserCardType.L: [],
                            UserCardType.XL: [], UserCardType.BUG: [], UserCardType.TECH_DEBT: []}
        self.colors_for_use = [UserCardColor.BLUE, UserCardColor.GREEN, UserCardColor.ORANGE,
                               UserCardColor.PINK, UserCardColor.PURPLE, UserCardColor.RED,
                               UserCardColor.YELLOW]

    def get_unused_color(self, uc_type: UserCardType, random_generator: Random):
        if len(self.used_colors[uc_type]) == 7:
            print("Не осталось не использованных цветов.")
            return
        # todo в Godot'е и python используются разные генераторы (псевдо-)случайных чисел
        cfu = copy(self.colors_for_use)
        for i in self.used_colors[uc_type]:
            cfu.remove(i)
        i = random_generator.randint(0, len(cfu) - 1)
        color = cfu[i]
        self.used_colors[uc_type].append(color)
        return color

    def release_color(self, us_type: UserCardType, color: UserCardColor):
        self.used_colors[us_type].remove(color)
