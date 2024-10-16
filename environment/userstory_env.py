from typing import List
from numpy.random import Generator

from game.common_methods import sample_n_or_zero
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo
from game.userstories.userstories import UserStoryCard
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from environment.card_methods import split_cards_in_types
from game.userstories.userstories import UserStories

USERSTORY_COMMON_FEATURE_COUNT = 4
USERSTORY_BUG_FEATURE_COUNT = 2
USERSTORY_TECH_DEBT_FEATURE_COUNT = 1


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

        self.max_action_num = + \
            self.us_common_count + \
            self.us_bug_count + \
            self.us_td_count

        self.userstories_common: List[UserStoryCard] = []
        self.userstories_bugs = []
        self.userstories_td = []

    def encode(self, userstories: UserStories, card_picker_random_generator: Generator):
        return self._encode_queue(userstories,
                                  self.us_common_count,
                                  self.us_bug_count,
                                  self.us_td_count,
                                  card_picker_random_generator)
    
    def get_encoded_card(self, index: int):
        # returns card by index
        if 0 <= index < len(self.userstories_common):
            return self.userstories_common[index]
        index -= self.us_common_count
        if 0 <= index < len(self.userstories_bugs):
            return self.userstories_bugs[index]
        index -= self.us_bug_count
        if 0 <= index < len(self.userstories_td):
            return self.userstories_td[index]
        return None

    def _encode_queue(self, userstories: UserStories, count_common, count_bug, count_td,
                      card_picker_random_generator: Generator):
        commons, bugs, tech_debts = split_cards_in_types(userstories.stories_list)

        sampled_cards_common = sample_n_or_zero(commons, count_common,
                                                card_picker_random_generator)
        sampled_cards_bugs = sample_n_or_zero(bugs, count_bug,
                                              card_picker_random_generator)
        sampled_cards_td = sample_n_or_zero(tech_debts, count_td,
                                            card_picker_random_generator)

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
