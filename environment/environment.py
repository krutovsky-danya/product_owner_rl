from game.game import ProductOwnerGame
from game.game_constants import UserCardType
import torch
import numpy as np
import random

from game.backlog_card.card_info import CardInfo
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo
from game.userstory_card.userstory_card_info import UserStoryCardInfo

BUG = UserCardType.BUG
TECH_DEBT = UserCardType.TECH_DEBT

USERSTORY_COMMON_FEATURE_COUNT = 4
USERSTORY_BUG_FEATURE_COUNT = 2
USERSTORY_TECH_DEBT_FEATURE_COUNT = 1

BACKLOG_COMMON_FEATURE_COUNT = 3
BACKLOG_BUG_FEATURE_COUNT = 2
BACKLOG_TECH_DEBT_FEATURE_COUNT = 2


class ProductOwnerEnv:
    def __init__(self, count_common_cards=4, count_bug_cards=2, count_td_cards=1,
                 count_common_userstories=4, count_bug_userstories=2, count_td_userstories=1):
        self.game = ProductOwnerGame()
        self.count_common_cards = count_common_cards
        self.count_bug_cards = count_bug_cards
        self.count_td_cards = count_td_cards
        self.count_common_us = count_common_userstories
        self.count_bug_us = count_bug_userstories
        self.count_td_us = count_td_userstories

        self.sampled_cards_common = None
        self.sampled_cards_bugs = None
        self.sampled_cards_td = None
        self.sampled_userstories_common = None
        self.sampled_userstories_bugs = None
        self.sampled_userstories_td = None
        self.current_state = self._get_state()

    def reset(self):
        self.game = ProductOwnerGame()
        self.current_state = self._get_state()
        return self.current_state

    def _get_state(self, in_tensor=True):
        context = self.game.context
        state = [context.current_sprint, context.get_money() / 10 ** 5,
                 context.customers, context.get_loyalty(), context.credit / 10 ** 5,
                 context.available_developers_count, context.current_rooms_counter,
                 context.current_sprint_hours, *self._get_completed_cards_count(),
                 *self._get_userstories_descriptions(), *self._get_backlog_cards_descriptions()]
        if in_tensor:
            return torch.tensor(state)
        else:
            return np.array(state)

    def _get_completed_cards_count(self):
        completed_cards = self.game.completed_us
        completed_us_count, completed_bug_count, completed_td_count = 0, 0, 0
        for card_info in completed_cards:
            if card_info.card_type == BUG:
                completed_bug_count += 1
            elif card_info.card_type == TECH_DEBT:
                completed_td_count += 1
            else:
                completed_us_count += 1
        return completed_us_count, completed_bug_count, completed_td_count

    def _get_userstories_descriptions(self):
        return self._get_cards_descriptions(self.count_common_us, self.count_bug_us,
                                            self.count_td_us, is_backlog=False)

    def _get_backlog_cards_descriptions(self):
        return self._get_cards_descriptions(self.count_common_cards, self.count_bug_cards,
                                            self.count_td_cards, is_backlog=True)

    def _get_cards_descriptions(self, count_common, count_bug, count_td, is_backlog):
        cards = self.game.backlog.backlog if is_backlog else self.game.userstories.stories_list
        commons, bugs, tech_debts = self._split_cards_in_types(cards)

        sampled_cards_common = random.sample(commons, min(count_common, len(commons)))
        sampled_cards_bugs = random.sample(bugs, min(count_bug, len(bugs)))
        sampled_cards_td = random.sample(tech_debts, min(count_td, len(tech_debts)))

        self._set_sampled_cards(sampled_cards_common, sampled_cards_bugs,
                                sampled_cards_td, is_backlog)

        return self._get_transforms_to_descriptions(sampled_cards_common, sampled_cards_bugs,
                                                    sampled_cards_td, is_backlog)

    def _split_cards_in_types(self, cards):
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

    def _set_sampled_cards(self, common, bugs, tech_debt, is_backlog):
        if is_backlog:
            self.sampled_cards_common = common
            self.sampled_cards_bugs = bugs
            self.sampled_cards_td = tech_debt
        else:
            self.sampled_userstories_common = common
            self.sampled_userstories_bugs = bugs
            self.sampled_userstories_td = tech_debt

    def _get_transforms_to_descriptions(self, sampled_cards_common, sampled_cards_bugs, sampled_cards_td,
                                        is_backlog):
        if is_backlog:
            description_common = self._get_transforms_to_descriptions_backlog_common(sampled_cards_common)
            description_bugs = self._get_transforms_to_descriptions_backlog_bug(sampled_cards_bugs)
            description_tech_debts = self._get_transforms_to_descriptions_backlog_tech_debt(sampled_cards_td)
        else:
            description_common = self._get_transforms_to_descriptions_userstory_common(sampled_cards_common)
            description_bugs = self._get_transforms_to_descriptions_userstory_bug(sampled_cards_bugs)
            description_tech_debts = self._get_transforms_to_descriptions_userstory_tech_debt(sampled_cards_td)

        return description_common + description_bugs + description_tech_debts

    def _get_transforms_to_descriptions_backlog_common(self, cards):
        res = [0] * self.count_common_cards * BACKLOG_COMMON_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: CardInfo = cards[i].info
            res[BACKLOG_COMMON_FEATURE_COUNT * i] = card_info.base_hours
            res[BACKLOG_COMMON_FEATURE_COUNT * i + 1] = card_info.hours
            res[BACKLOG_COMMON_FEATURE_COUNT * i + 2] = card_info.card_type.value

        return res

    def _get_transforms_to_descriptions_backlog_bug(self, cards):
        res = [0] * self.count_bug_cards * BACKLOG_BUG_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: CardInfo = cards[i].info
            res[BACKLOG_BUG_FEATURE_COUNT * i] = card_info.base_hours
            res[BACKLOG_BUG_FEATURE_COUNT * i + 1] = card_info.hours

        return res

    def _get_transforms_to_descriptions_backlog_tech_debt(self, cards):
        res = [0] * self.count_td_cards * BACKLOG_TECH_DEBT_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: CardInfo = cards[i].info
            res[BACKLOG_BUG_FEATURE_COUNT * i] = card_info.base_hours
            res[BACKLOG_BUG_FEATURE_COUNT * i + 1] = card_info.hours

        return res

    def _get_transforms_to_descriptions_userstory_common(self, cards):
        res = [0] * self.count_common_us * USERSTORY_COMMON_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: UserStoryCardInfo = cards[i].info
            res[USERSTORY_COMMON_FEATURE_COUNT * i] = card_info.customers_to_bring
            res[USERSTORY_COMMON_FEATURE_COUNT * i + 1] = card_info.loyalty
            res[USERSTORY_COMMON_FEATURE_COUNT * i + 2] = card_info.spawn_sprint
            res[USERSTORY_COMMON_FEATURE_COUNT * i + 3] = card_info.card_type.value

        return res

    def _get_transforms_to_descriptions_userstory_bug(self, cards):
        res = [0] * self.count_bug_us * USERSTORY_BUG_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: BugUserStoryInfo = cards[i].info
            res[USERSTORY_BUG_FEATURE_COUNT * i] = card_info.loyalty_debuff
            res[USERSTORY_BUG_FEATURE_COUNT * i + 1] = card_info.customers_debuff

        return res

    def _get_transforms_to_descriptions_userstory_tech_debt(self, cards):
        res = [0] * self.count_td_us * USERSTORY_TECH_DEBT_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: TechDebtInfo = cards[i].info
            res[USERSTORY_TECH_DEBT_FEATURE_COUNT * i] = card_info.full_hours_debuff

        return res

    def step(self, action: int):
        # new_state, reward, done, info
        self._perform_action(action)
        reward = self._get_reward()
        self.current_state = self._get_state()
        return self.current_state, reward, self.game.context.done, None

    def _get_reward(self):
        sprint_penalty = -1
        money_reward = self.game.context.get_money() / 10 ** 6 - 1
        return sprint_penalty + money_reward

    def _perform_action(self, action: int):
        # we'll assume that action in range(0, max_action_num)
        if action == 0:
            self.game.backlog_start_sprint()
        elif action == 1:
            self.game.userstories_start_release()
        elif action == 2:
            self.game.hud_release_product()
        elif action == 3:
            room_num = self._get_min_not_full_room_number()
            self.game.buy_robot(room_num)
        elif action == 4:
            room_num = self._get_min_available_to_buy_room_number()
            self.game.buy_room(room_num)
        elif action == 5:
            self.game.press_statistical_research()
        elif action == 6:
            self.game.press_user_survey()
        else:
            self._perform_action_card(action - 7)

    def _perform_action_card(self, action: int):
        userstory_max_action_num = self.count_common_us + self.count_bug_us + self.count_td_us
        if action < userstory_max_action_num:
            self._perform_action_userstory(action)
        else:
            self._perform_action_backlog_card(action - userstory_max_action_num)

    def _perform_action_backlog_card(self, action: int):
        if action < self.count_common_cards:
            card = self.sampled_cards_common[action]
        elif action - self.count_common_cards < self.count_bug_cards:
            card = self.sampled_cards_bugs[action - self.count_common_cards]
        else:
            card = self.sampled_cards_td[action - self.count_common_cards - self.count_bug_cards]
        self.game.move_backlog_card(card)

    def _perform_action_userstory(self, action: int):
        if action < self.count_common_us:
            card = self.sampled_userstories_common[action]
        elif action - self.count_common_us < self.count_bug_us:
            card = self.sampled_userstories_bugs[action - self.count_common_us]
        else:
            card = self.sampled_userstories_td[action - self.count_common_us - self.count_bug_us]
        self.game.move_userstory_card(card)

    def _get_min_not_full_room_number(self):
        offices = self.game.office.offices
        for i in range(len(offices)):
            room = offices[i]
            if room.can_buy_robot:
                return i
        return -1

    def _get_min_available_to_buy_room_number(self):
        offices = self.game.office.offices
        for i in range(len(offices)):
            room = offices[i]
            if room.can_buy_room:
                return i
        return -1


if __name__ == "__main__":
    g = [(12, 14, 1), 100]
    f = [12, 13, 6, 7, *g]
    print(10 ** 5)
