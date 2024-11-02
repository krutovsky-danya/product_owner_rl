from game.backlog.backlog import Backlog
from game.backlog_card.backlog_card import Card
from game.backlog_card.card_info import CardInfo
from game.common_methods import sample_n_or_zero
from environment.card_methods import split_cards_in_types
from numpy.random import Generator


from typing import List, Tuple, Optional

from game.game_variables import GlobalContext
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo

BACKLOG_COMMON_FEATURE_COUNT = 7
BACKLOG_BUG_FEATURE_COUNT = 5
BACKLOG_TECH_DEBT_FEATURE_COUNT = 4


class BacklogEnv:
    def __init__(self, backlog_commons_count=12, backlog_bugs_count=2, backlog_tech_debt_count=2,
                 sprint_commons_count=12, sprint_bugs_count=2, sprint_tech_debt_count=2) -> None:
        self.backlog_commons_count = backlog_commons_count
        self.backlog_bugs_count = backlog_bugs_count
        self.backlog_tech_debt_count = backlog_tech_debt_count

        self.sprint_commons_count = sprint_commons_count
        self.sprint_bugs_count = sprint_bugs_count
        self.sprint_tech_debt_count = sprint_tech_debt_count

        self.backlog_space_dim = self.backlog_commons_count * BACKLOG_COMMON_FEATURE_COUNT + \
            self.backlog_bugs_count * BACKLOG_BUG_FEATURE_COUNT + \
            self.backlog_tech_debt_count * BACKLOG_TECH_DEBT_FEATURE_COUNT

        self.sprint_space_dim = self.sprint_commons_count * BACKLOG_COMMON_FEATURE_COUNT + \
            self.sprint_bugs_count * BACKLOG_BUG_FEATURE_COUNT + \
            self.sprint_tech_debt_count * BACKLOG_TECH_DEBT_FEATURE_COUNT

        self.backlog_max_action_num = + \
            self.backlog_commons_count + \
            self.backlog_bugs_count + \
            self.backlog_tech_debt_count

        self.sprint_max_action_num = + \
            self.sprint_commons_count + \
            self.sprint_bugs_count + \
            self.sprint_tech_debt_count

        self.backlog: Optional[Backlog] = None
        self.context: Optional[GlobalContext] = None

        self.backlog_commons: List[Card] = []
        self.backlog_bugs: List[Card] = []
        self.backlog_tech_debt: List[Card] = []

        self.sprint_commons: List[Card] = []
        self.sprint_bugs: List[Card] = []
        self.sprint_tech_debt: List[Card] = []

    def _set_backlog_cards(self, commons, bugs, tech_debt):
        self.backlog_commons = commons
        self.backlog_bugs = bugs
        self.backlog_tech_debt = tech_debt

    def _set_sprint_cards(self, commons, bugs, tech_debt):
        self.sprint_commons = commons
        self.sprint_bugs = bugs
        self.sprint_tech_debt = tech_debt

    def get_card(self, index: int):
        if 0 <= index < len(self.backlog_commons):
            return self.backlog_commons[index]

        bug_card_id = index - self.backlog_commons_count
        if 0 <= bug_card_id < len(self.backlog_bugs):
            return self.backlog_bugs[bug_card_id]

        tech_debt_card_id = bug_card_id - self.backlog_bugs_count
        if 0 <= tech_debt_card_id < len(self.backlog_tech_debt):    
            return self.backlog_tech_debt[tech_debt_card_id]

    def encode(self, backlog: Backlog, card_picker_random_generator: Generator) -> List[float]:
        self.backlog = backlog
        self.context = backlog.context
        counts = (self.backlog_commons_count,
                  self.backlog_bugs_count, self.backlog_tech_debt_count)
        backlog_encoding = self._encode_queue(
            backlog.backlog, counts, self._set_backlog_cards, card_picker_random_generator)
        assert len(backlog_encoding) == self.backlog_space_dim

        sprint_encoding = []

        if self.sprint_space_dim > 0:
            counts = (self.sprint_commons_count, self.sprint_bugs_count,
                      self.sprint_tech_debt_count)
            sprint_encoding = self._encode_queue(
                backlog.sprint, counts, self._set_sprint_cards, card_picker_random_generator)
            assert len(sprint_encoding) == self.sprint_space_dim

        return backlog_encoding + sprint_encoding

    def _encode_queue(self, cards: List[Card], counts: Tuple[int, int, int], setter,
                      card_picker_random_generator: Generator) -> List[float]:
        commons, bugs, tech_debt = split_cards_in_types(cards)
        commons_count, bugs_count, tech_debt_count = counts

        commons = sample_n_or_zero(commons, commons_count,
                                   card_picker_random_generator)
        bugs = sample_n_or_zero(bugs, bugs_count,
                                card_picker_random_generator)
        tech_debt = sample_n_or_zero(tech_debt, tech_debt_count,
                                     card_picker_random_generator)
        setter(commons, bugs, tech_debt)

        commons_len = BACKLOG_COMMON_FEATURE_COUNT * commons_count
        commons_encoding = self._encode_cards(
            commons, self._encode_common_card, commons_len)

        bugs_len = BACKLOG_BUG_FEATURE_COUNT * bugs_count
        bugs_encoding = self._encode_cards(bugs, self._encode_bug, bugs_len)

        tech_debt_len = BACKLOG_TECH_DEBT_FEATURE_COUNT * tech_debt_count
        tech_debt_encoding = self._encode_cards(
            tech_debt, self._encode_tech_debt, tech_debt_len)

        encoding = commons_encoding + bugs_encoding + tech_debt_encoding

        return encoding

    def _encode_cards(self, cards, encoder, result_len) -> List[float]:
        result = []
        for card in cards:
            result.extend(encoder(card))
        padding = result_len - len(result)
        result.extend([0] * padding)
        return result

    def _encode_common_card(self, card: Card) -> List[float]:
        card_info = card.info
        us_card_info = self.context.current_stories[card_info.us_id]
        life_time = self.context.current_sprint - us_card_info.spawn_sprint
        encoded = [
            card_info.base_hours,
            card_info.hours,
            card_info.card_type.value,
            us_card_info.completed_part,
            us_card_info.loyalty,
            us_card_info.customers_to_bring,
            life_time,
        ]
        assert len(encoded) == BACKLOG_COMMON_FEATURE_COUNT
        return encoded

    def _encode_bug(self, card: Card) -> List[float]:
        card_info = card.info
        us_card_info: BugUserStoryInfo = self.context.current_stories[card_info.us_id]
        encoded = [
            card_info.base_hours,
            card_info.hours,
            us_card_info.completed_part,
            us_card_info.loyalty_debuff,
            us_card_info.customers_debuff,
        ]
        assert len(encoded) == BACKLOG_BUG_FEATURE_COUNT
        return encoded

    def _encode_tech_debt(self, card: Card) -> List[float]:
        card_info = card.info
        us_card_info: TechDebtInfo = self.context.current_stories[card_info.us_id]
        encoded = [
            card_info.base_hours,
            card_info.hours,
            us_card_info.completed_part,
            us_card_info.full_hours_debuff,
        ]
        assert len(encoded) == BACKLOG_TECH_DEBT_FEATURE_COUNT
        return encoded
