import random
from game import game_global as Global
from game.backlog_card.card_info import CardInfo


class UserStoryCardInfo:
    def __init__(self, label_val: str = ""):
        self.customers_to_bring = 0
        self.loyalty = 0
        self.time_to_complete = 0
        self.label = label_val
        self.completed = False
        self.is_decomposed = False
        self.completed_part = 0
        self.spawn_sprint = Global.current_sprint
        self.card_type = Global.UserCardType.S
        self._set_card_type(label_val)
        if not (self.card_type == Global.UserCardType.BUG or self.card_type == Global.UserCardType.TECH_DEBT):
            self._set_loyalty_and_customers_ordinary_us()
        self.color = Global.get_unused_color(self.card_type)
        self.related_cards = []
        self.generate_related_cards()

    def _set_card_type(self, label_val: str):
        if label_val == "S":
            self.time_to_complete = 38
            self.card_type = Global.UserCardType.S
        elif label_val == "M":
            self.time_to_complete = 76
            self.card_type = Global.UserCardType.M
        elif label_val == "L":
            self.time_to_complete = 152
            self.card_type = Global.UserCardType.L
        elif label_val == "XL":
            self.time_to_complete = 304
            self.card_type = Global.UserCardType.XL
        elif label_val == "Bug":
            self.time_to_complete = random.randint(1, 38)
            self.card_type = Global.UserCardType.BUG
        elif label_val == "TechDebt":
            self.time_to_complete = random.randint(1, 5)
            self.card_type = Global.UserCardType.TECH_DEBT

    def _set_loyalty_and_customers_ordinary_us(self):
        r_lty = random.uniform(Global.USERSTORY_LOYALTY[self.card_type][0], Global.USERSTORY_LOYALTY[self.card_type][1])
        self.loyalty = Global.stepify(r_lty, 0.005)
        r_user = random.uniform(Global.USERSTORY_CUSTOMER[self.card_type][0], Global.USERSTORY_CUSTOMER[self.card_type][1])
        self.customers_to_bring = Global.stepify(r_user, 0.5)

    def generate_related_cards(self):
        time = 0
        while time < self.time_to_complete:
            time_for_card = random.randint(6, 19)
            if time_for_card + time > self.time_to_complete:
                time_for_card = self.time_to_complete - time
            card = CardInfo(hours_val=time_for_card, color_val=self.color, us_id_val=id(self),
                            label_val=self.label, card_type_val=self.card_type)
            self.related_cards.append(card)
            time += time_for_card


if __name__ == "__main__":
    uss = UserStoryCardInfo("S")
    usm = UserStoryCardInfo("M")
    usl = UserStoryCardInfo("L")
    usxl = UserStoryCardInfo("XL")
    usbug = UserStoryCardInfo("Bug")
    ustd = UserStoryCardInfo("TechDebt")
    print(12)
