from game.common_methods import stepify
from game.game_constants import GlobalConstants, UserCardType
from game.game_colors import ColorStorage
from game.backlog_card.card_info import CardInfo
from random import Random

from typing import List


class UserStoryCardInfo:
    def __init__(
        self,
        label_val: str,
        spawn_sprint: int,
        color_storage: ColorStorage,
        random_generator: Random,
    ):
        self.customers_to_bring = 0
        self.loyalty = 0
        self.time_to_complete = 0
        self.label = label_val
        self.completed = False
        self.is_decomposed = False
        self.completed_part = 0
        self.spawn_sprint = spawn_sprint
        self.card_type = UserCardType.S
        self._set_card_type(label_val, random_generator)
        if not (
            self.card_type == UserCardType.BUG
            or self.card_type == UserCardType.TECH_DEBT
        ):
            self._set_loyalty_and_customers_ordinary_us(random_generator)
        self.color = color_storage.get_unused_color(self.card_type, random_generator)
        self.related_cards: List[CardInfo] = []
        self.generate_related_cards(random_generator)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, UserStoryCardInfo):
            return False
        if abs(self.loyalty - value.loyalty) > 1e-3:
            return False
        if abs(self.customers_to_bring - value.customers_to_bring) > 1e-3:
            return False
        return self.label == value.label

    def __repr__(self) -> str:
        return f"UserstoryCardInfo({self.label}, loyalty={self.loyalty}, customers={self.customers_to_bring}, color={self.color}, spawn_sprint={self.spawn_sprint})"

    def _set_card_type(self, label_val: str, random_generator: Random):
        if label_val == "S":
            self.time_to_complete = 38
            self.card_type = UserCardType.S
        elif label_val == "M":
            self.time_to_complete = 76
            self.card_type = UserCardType.M
        elif label_val == "L":
            self.time_to_complete = 152
            self.card_type = UserCardType.L
        elif label_val == "XL":
            self.time_to_complete = 304
            self.card_type = UserCardType.XL
        elif label_val == "Bug":
            self.time_to_complete = random_generator.randint(1, 38)
            self.card_type = UserCardType.BUG
        elif label_val == "TechDebt":
            self.time_to_complete = random_generator.randint(1, 5)
            self.card_type = UserCardType.TECH_DEBT

    def _set_loyalty_and_customers_ordinary_us(self, random_generator: Random):
        user_story_loyalty = GlobalConstants.USERSTORY_LOYALTY
        user_story_customers = GlobalConstants.USERSTORY_CUSTOMER
        min_lty, max_lty = user_story_loyalty[self.card_type]
        r_lty = random_generator.uniform(min_lty, max_lty)
        self.loyalty = stepify(r_lty, 0.005)
        min_user, max_user = user_story_customers[self.card_type]
        r_user = random_generator.uniform(min_user, max_user)
        self.customers_to_bring = stepify(r_user, 0.5)

    def generate_related_cards(self, random_generator: Random):
        time = 0
        while time < self.time_to_complete:
            time_for_card = random_generator.randint(6, 19)
            if time_for_card + time > self.time_to_complete:
                time_for_card = self.time_to_complete - time
            card = CardInfo(
                hours_val=time_for_card,
                color_val=self.color,
                us_id_val=id(self),
                label_val=self.label,
                card_type_val=self.card_type,
            )
            self.related_cards.append(card)
            time += time_for_card
