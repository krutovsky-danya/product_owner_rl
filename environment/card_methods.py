from typing import List

from game.backlog_card.backlog_card import Card
from game.game_constants import UserCardType

BUG = UserCardType.BUG
TECH_DEBT = UserCardType.TECH_DEBT


def split_cards_in_types(cards: List[Card]):
    commons = []
    bugs = []
    tech_debts = []
    for card in cards:
        card_info = card.info
        if card_info.card_type == BUG:
            bugs.append(card)
        elif card_info.card_type == TECH_DEBT:
            tech_debts.append(card)
        else:
            commons.append(card)

    return commons, bugs, tech_debts
