from game.game_colors import ColorStorage
from game.userstory_card.userstory_card import UserStoryCard
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from random import Random


class UserStoriesGenerator:
    def __init__(self, s: int, m: int, l: int, xl: int, random_gen: Random):
        self.card_types = []
        sizes_count = [[s, "S"], [m, "M"], [l, "L"], [xl, "XL"]]
        for size in sizes_count:
            self.card_types += [size[1]] * size[0]
        self.random_gen = random_gen

    def generate_userstories(self, count: int, spawn_sprint: int, color_storage: ColorStorage):
        result = []

        for i in range(count):
            card_type = self.random_gen.choice(self.card_types)
            card = UserStoryCard(UserStoryCardInfo(card_type, 
                                                   spawn_sprint, 
                                                   color_storage,
                                                   self.random_gen))

            result.append(card)

        return result
