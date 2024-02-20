from typing import List
from game.backlog_card.backlog_card import Card
from game.common_methods import sample_n_or_zero
from game.game_constants import UserCardType
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo
from game.userstory_card.userstory_card_info import UserStoryCardInfo

BUG = UserCardType.BUG
TECH_DEBT = UserCardType.TECH_DEBT

USERSTORY_COMMON_FEATURE_COUNT = 4
USERSTORY_BUG_FEATURE_COUNT = 2
USERSTORY_TECH_DEBT_FEATURE_COUNT = 1


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


class UserstoryEnv:
    def __init__(self,
                 userstories_common_count=6,
                 userstories_bug_count=2,
                 userstories_td_count=1) -> None:
        self.us_common_count = userstories_common_count
        self.us_bug_count = userstories_bug_count
        self.us_td_count = userstories_td_count

        self.userstory_space_dim = + \
            self.us_common_count * USERSTORY_COMMON_FEATURE_COUNT + \
            self.us_bug_count * USERSTORY_BUG_FEATURE_COUNT + \
            self.us_td_count * USERSTORY_TECH_DEBT_FEATURE_COUNT

        self.userstories_common = []
        self.userstories_bugs = []
        self.userstories_td = []

    def encode(self, cards):
        return self._encode_queue(cards,
                                  self.us_common_count,
                                  self.us_bug_count,
                                  self.us_td_count)

    def _encode_queue(self, cards, count_common, count_bug, count_td):
        commons, bugs, tech_debts = split_cards_in_types(cards)

        sampled_cards_common = sample_n_or_zero(commons, count_common)
        sampled_cards_bugs = sample_n_or_zero(bugs, count_bug)
        sampled_cards_td = sample_n_or_zero(tech_debts, count_td)

        self._set_sampled_cards(sampled_cards_common, sampled_cards_bugs,
                                sampled_cards_td)

        return self._encode(sampled_cards_common, sampled_cards_bugs,
                            sampled_cards_td)

    def _set_sampled_cards(self, common, bugs, tech_debt):
        self.userstories_common = common
        self.userstories_bugs = bugs
        self.userstories_td = tech_debt

    def _encode(self, sampled_cards_common, sampled_cards_bugs, sampled_cards_td):
        description_common = self._encode_userstory_common(sampled_cards_common)
        description_bugs = self._encode_userstory_bug(sampled_cards_bugs)
        description_tech_debts = self._encode_userstory_tech_debt(sampled_cards_td)

        encoding = description_common + description_bugs + description_tech_debts

        assert len(encoding) == self.userstory_space_dim

        return encoding

    def _encode_userstory_common(self, cards):
        res = [0] * self.us_common_count * USERSTORY_COMMON_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: UserStoryCardInfo = cards[i].info
            start = USERSTORY_COMMON_FEATURE_COUNT * i
            end = USERSTORY_COMMON_FEATURE_COUNT * (i + 1)
            res[start:end] = [card_info.customers_to_bring,
                              card_info.loyalty,
                              card_info.spawn_sprint,
                              card_info.card_type.value]

        return res

    def _encode_userstory_bug(self, cards):
        res = [0] * self.us_bug_count * USERSTORY_BUG_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: BugUserStoryInfo = cards[i].info
            start = USERSTORY_BUG_FEATURE_COUNT * i
            end = USERSTORY_BUG_FEATURE_COUNT * (i + 1)
            res[start:end] = [card_info.loyalty_debuff,
                              card_info.customers_debuff]

        return res

    def _encode_userstory_tech_debt(self, cards):
        res = [0] * self.us_td_count * USERSTORY_TECH_DEBT_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: TechDebtInfo = cards[i].info
            start = USERSTORY_TECH_DEBT_FEATURE_COUNT * i
            end = USERSTORY_TECH_DEBT_FEATURE_COUNT * (i + 1)
            res[start:end] = [card_info.full_hours_debuff]

        return res
