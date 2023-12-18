from game.game import ProductOwnerGame
from game.game_constants import UserCardType
import torch
import numpy as np
import random
from typing import List

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

        self.state_dim = 16 + \
            self.count_common_cards * BACKLOG_COMMON_FEATURE_COUNT + \
            self.count_bug_cards * BACKLOG_BUG_FEATURE_COUNT + \
            self.count_td_cards * BACKLOG_TECH_DEBT_FEATURE_COUNT + \
            self.count_common_us * USERSTORY_COMMON_FEATURE_COUNT + \
            self.count_bug_us * USERSTORY_BUG_FEATURE_COUNT + \
            self.count_td_us * USERSTORY_TECH_DEBT_FEATURE_COUNT
        
        self.current_state = self._get_state()

        self.userstory_max_action_num = self.count_common_us + \
            self.count_bug_us + self.count_td_us
        self.action_n = 7 + \
            self.userstory_max_action_num + \
            self.count_common_cards + self.count_bug_cards + self.count_td_cards

    def reset(self):
        self.game = ProductOwnerGame()
        self.current_state = self._get_state()
        return self.current_state

    def _get_state(self, in_tensor=False):
        context = self.game.context
        state = [
            context.current_sprint,
            context.get_money() / 10 ** 5,
            context.customers,
            context.get_loyalty(),
            context.credit / 10 ** 5,
            context.available_developers_count,
            context.current_rooms_counter,
            context.current_sprint_hours,
            self.game.backlog.can_start_sprint(),
            self.game.userstories.release_available,
            self.game.hud.release_available,
            self.game.userstories.statistical_research_available,
            self.game.userstories.user_survey_available,
            *self._get_completed_cards_count(),
            *self._get_userstories_descriptions(),
            *self._get_backlog_cards_descriptions()
        ]
        assert len(state) == self.state_dim
        if in_tensor:
            return torch.tensor(state)
        else:
            return np.array(state, dtype=np.float32)

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

        sampled_cards_common = random.sample(
            commons, min(count_common, len(commons)))
        sampled_cards_bugs = random.sample(bugs, min(count_bug, len(bugs)))
        sampled_cards_td = random.sample(
            tech_debts, min(count_td, len(tech_debts)))

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
            description_common = self._get_transforms_to_descriptions_backlog_common(
                sampled_cards_common)
            description_bugs = self._get_transforms_to_descriptions_backlog_bug(
                sampled_cards_bugs)
            description_tech_debts = self._get_transforms_to_descriptions_backlog_tech_debt(
                sampled_cards_td)
        else:
            description_common = self._get_transforms_to_descriptions_userstory_common(
                sampled_cards_common)
            description_bugs = self._get_transforms_to_descriptions_userstory_bug(
                sampled_cards_bugs)
            description_tech_debts = self._get_transforms_to_descriptions_userstory_tech_debt(
                sampled_cards_td)

        return description_common + description_bugs + description_tech_debts

    def _get_transforms_to_descriptions_backlog_common(self, cards):
        res = [0] * self.count_common_cards * BACKLOG_COMMON_FEATURE_COUNT

        for i in range(len(cards)):
            card_info: CardInfo = cards[i].info
            res[BACKLOG_COMMON_FEATURE_COUNT * i] = card_info.base_hours
            res[BACKLOG_COMMON_FEATURE_COUNT * i + 1] = card_info.hours
            res[BACKLOG_COMMON_FEATURE_COUNT *
                i + 2] = card_info.card_type.value

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
            res[USERSTORY_COMMON_FEATURE_COUNT *
                i] = card_info.customers_to_bring
            res[USERSTORY_COMMON_FEATURE_COUNT * i + 1] = card_info.loyalty
            res[USERSTORY_COMMON_FEATURE_COUNT * i + 2] = card_info.spawn_sprint
            res[USERSTORY_COMMON_FEATURE_COUNT *
                i + 3] = card_info.card_type.value

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
            res[USERSTORY_TECH_DEBT_FEATURE_COUNT *
                i] = card_info.full_hours_debuff

        return res

    def step(self, action: int):
        # new_state, reward, done, info
        reward = 0
        credit_before = self.game.context.credit
        reward_bit = self._perform_action(action)
        credit_after = self.game.context.credit
        if credit_before > 0 and credit_after <= 0:
            print('Credit paid')
            reward += 100
        reward += self._get_reward()
        reward += reward_bit
        self.current_state = self._get_state()
        return self.current_state, reward, self.game.context.done, None

    def _get_reward(self):
        # sprint_penalty = +1
        # money_reward = self.game.context.get_money() / 10 ** 6
        done = self.game.context.done
        if done:
            if self.game.context.get_money() > 1e6:
                reward_for_endgame = 500
            else:
                reward_for_endgame = -500
        else:
            reward_for_endgame = 0
        return reward_for_endgame

    def _perform_start_sprint_action(self) -> int:
        self.game.backlog_start_sprint()
        return 1

    def _perform_decomposition(self) -> int:
        is_release_available = self.game.userstories.release_available
        self.game.userstories_start_release()
        if is_release_available:
            return 1
        return -10
    
    def _perform_release(self) -> int:
        is_release_available = self.game.hud.release_available
        self.game.hud_release_product()
        if is_release_available:
            return 1
        return -10

    def _perform_buy_robot(self) -> int:
        room_num = self._get_min_not_full_room_number()
        if room_num == -1:
            return -10
        worker_count_before = self.game.context.available_developers_count
        self.game.buy_robot(room_num)
        worker_count = self.game.context.available_developers_count
        if worker_count_before == worker_count:
            return -10
        return 1
    
    def _perform_buy_room(self) -> int:
        room_num = self._get_min_available_to_buy_room_number()
        if room_num == -1:
            return -10
        worker_count_before = self.game.context.available_developers_count
        self.game.buy_room(room_num)
        worker_count = self.game.context.available_developers_count
        if worker_count_before == worker_count:
            return -10
        return 1
    
    def _perform_get_statistical_research(self) -> int:
        if not self.game.userstories.statistical_research_available:
            return -10
        stories_before = len(self.game.userstories.stories_list)
        self.game.press_statistical_research()
        stories_after = len(self.game.userstories.stories_list)
        if stories_before == stories_after:
            return -10
        return 1
    
    def _perform_user_survey(self) -> int:
        if not self.game.userstories.user_survey_available:
            return -10
        stories_before = len(self.game.userstories.stories_list)
        self.game.press_user_survey()
        stories_after = len(self.game.userstories.stories_list)
        if stories_before == stories_after:
            return -10
        return 1
    
    def _perform_action(self, action: int):
        # we'll assume that action in range(0, max_action_num)
        if action == 0:
            return self._perform_start_sprint_action()
        if action == 1:
            return self._perform_decomposition()
        if action == 2:
            return self._perform_release()
        if action == 3:
            return self._perform_buy_robot()
        if action == 4:
            return self._perform_buy_room()
        if action == 5:
            return self._perform_get_statistical_research()
        if action == 6:
            return self._perform_user_survey()
        
        return self._perform_action_card(action - 7)

    def _perform_action_card(self, action: int) -> int:
        if action < self.userstory_max_action_num:
            return self._perform_action_userstory(action)
        
        card_id = action - self.userstory_max_action_num
        return self._perform_action_backlog_card(card_id)

    def _perform_action_backlog_card(self, action: int) -> int:
        if action < self.count_common_cards:
            card = self._get_card(self.sampled_cards_common, action)
        elif action - self.count_common_cards < self.count_bug_cards:
            card = self._get_card(self.sampled_cards_bugs,
                                  action - self.count_common_cards)
        else:
            card = self._get_card(
                self.sampled_cards_td, action - self.count_common_cards - self.count_bug_cards)
        if card is not None:
            self.game.move_backlog_card(card)
            return 1
        return -10

    def _perform_action_userstory(self, action: int) -> int:
        if action < self.count_common_us:
            card = self._get_card(self.sampled_userstories_common, action)
        elif action - self.count_common_us < self.count_bug_us:
            card = self._get_card(
                self.sampled_userstories_bugs, action - self.count_common_us)
        else:
            card = self._get_card(
                self.sampled_userstories_td, action - self.count_common_us - self.count_bug_us)
        if card is not None:
            self.game.move_userstory_card(card)
            return 1
        return -10

    def _get_card(self, sampled, index):
        if index < len(sampled):
            return sampled[index]
        return None

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
    print(1e6)
