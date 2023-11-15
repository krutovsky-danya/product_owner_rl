from game.userstory_card.userstory_card import UserStoryCard
from game.userstory_card.userstory_card_info import UserStoryCardInfo
import random


class UserStoriesGenerator:
    def __init__(self, s: int, m: int, l: int, xl: int):
        self.a = []
        sizes_count = [[s, "S"], [m, "M"], [l, "L"], [xl, "XL"]]
        for size in sizes_count:
            self.a += [size[1]] * size[0]

    def generate_userstories(self, count: int):
        result = []

        for i in range(count):
            card_type = self.a[random.randint(0, len(self.a) - 1)]
            card = UserStoryCard(UserStoryCardInfo(label_val=card_type))

            result.append(card)

        return result


if __name__ == "__main__":
    usg = UserStoriesGenerator(100, 0, 0, 0)
    gen = usg.generate_userstories(2)
    print(gen)
    usg1 = UserStoriesGenerator(1, 59, 30, 10)
    gen1 = usg1.generate_userstories(2)
    print(gen1)
    print(12)
