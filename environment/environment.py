from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from .reward_sytem import BaseRewardSystem
from game.backlog_card.backlog_card import Card
from game.game import ProductOwnerGame
from game.game_constants import UserCardType
import torch
import numpy as np
from game.game_generators import get_buggy_game_1

BUG = UserCardType.BUG
TECH_DEBT = UserCardType.TECH_DEBT

START_SPRINT = 0
DECOMPOSE = 1
RELEASE = 2
BUY_ROBOT = 3
BUY_ROOM = 4
STATISTICAL_RESEARCH = 5
USER_SURVEY = 6


class ProductOwnerEnv:
    IS_SILENT = False

    def __init__(self, userstory_env=None, backlog_env: BacklogEnv = None, with_info=True,
                 reward_system: BaseRewardSystem = None,
                 seed=None, card_picker_seed=None):
        self.game = ProductOwnerGame(seed=seed)
        if backlog_env is None:
            self.backlog_env = BacklogEnv()
        else:
            self.backlog_env = backlog_env
        self.userstory_env = UserstoryEnv() if userstory_env is None else userstory_env
        self.card_picker_random_gen = np.random.default_rng(seed=card_picker_seed)

        self.meta_space_dim = 19

        self.state_dim = self.meta_space_dim + \
            self.userstory_env.userstory_space_dim + \
            self.backlog_env.backlog_space_dim + \
            self.backlog_env.sprint_space_dim
        
        self.current_state = self._get_state()

        self.meta_action_dim = 7

        self.action_n = self.meta_action_dim + \
            self.userstory_env.max_action_num + \
            self.backlog_env.backlog_max_action_num + \
            self.backlog_env.sprint_max_action_num

        self.with_info = with_info
        if reward_system is None:
            raise Exception("reward system can not be None")
        self.reward_system = reward_system

    def reset(self, seed=None, card_picker_seed=None):
        self.game = ProductOwnerGame(seed=seed)
        self._reset_card_picker_random_gen(card_picker_seed)
        self.current_state = self._get_state()
        return self.current_state

    def _reset_card_picker_random_gen(self, card_picker_seed=None):
        self.card_picker_random_gen = np.random.default_rng(seed=card_picker_seed)

    def recalculate_state(self):
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
            self.game.backlog.calculate_hours_sum(),
            context.blank_sprint_counter,
            self.game.backlog.can_start_sprint(),
            self.game.hud.release_available,
            self.game.userstories.release_available,
            self.game.userstories.statistical_research_available,
            self.game.userstories.user_survey_available,
            int(context.done),
            *self._get_completed_cards_count(),
            *self.userstory_env.encode(self.game.userstories, self.card_picker_random_gen),
            *self.backlog_env.encode(self.game.backlog, self.card_picker_random_gen)
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

    def get_info(self):
        if self.with_info:
            result = self._get_info_meta_actions()
            result += self._get_info_cards()
        else:
            result = list(range(self.action_n))
        return {"actions": result}

    def _get_info_meta_actions(self):
        result = []
        if self.game.is_backlog_start_sprint_available():
            result.append(START_SPRINT)
        if self.game.is_userstories_start_release_available():
            result.append(DECOMPOSE)
        if self.game.is_hud_release_product_available():
            result.append(RELEASE)
        if self.game.is_buy_robot_available():
            result.append(BUY_ROBOT)
        if self.game.is_buy_room_available():
            result.append(BUY_ROOM)
        if self.game.is_press_statistical_research_available():
            result.append(STATISTICAL_RESEARCH)
        if self.game.is_press_user_survey_available():
            result.append(USER_SURVEY)
        return result

    def _get_info_cards(self):
        result = self._get_info_userstory_cards()
        result += self._get_info_backlog_cards()
        result += self._get_info_sprint_cards()
        return result

    def _get_info_userstory_cards(self):
        result = []
        predicate = self.game.is_move_userstory_card_available
        offset = self.meta_action_dim
        self._set_info_cards(self.userstory_env.userstories_common, offset, predicate, result)
        offset += self.userstory_env.us_common_count
        self._set_info_cards(self.userstory_env.userstories_bugs, offset, predicate, result)
        offset += self.userstory_env.us_bug_count
        self._set_info_cards(self.userstory_env.userstories_td, offset, predicate, result)
        return result

    def _get_info_backlog_cards(self):
        result = []
        predicate = self.game.is_move_backlog_card_available
        offset = self.meta_action_dim + self.userstory_env.max_action_num
        self._set_info_cards(self.backlog_env.backlog_commons, offset, predicate, result)
        offset += self.backlog_env.backlog_commons_count
        self._set_info_cards(self.backlog_env.backlog_bugs, offset, predicate, result)
        offset += self.backlog_env.backlog_bugs_count
        self._set_info_cards(self.backlog_env.backlog_tech_debt, offset, predicate, result)
        return result

    def _get_info_sprint_cards(self):
        result = []
        predicate = self.game.is_move_sprint_card_available
        offset = self.meta_action_dim + self.userstory_env.max_action_num + self.backlog_env.backlog_max_action_num
        self._set_info_cards(self.backlog_env.sprint_commons, offset, predicate, result)
        offset += self.backlog_env.sprint_commons_count
        self._set_info_cards(self.backlog_env.sprint_bugs, offset, predicate, result)
        offset += self.backlog_env.sprint_bugs_count
        self._set_info_cards(self.backlog_env.sprint_tech_debt, offset, predicate, result)
        return result

    def _set_info_cards(self, cards, offset: int, predicate, result):
        for value, card in enumerate(cards):
            if predicate(card):
                result.append(value + offset)

    def step(self, action: int):
        # new_state, reward, done, info
        state_old = self.reward_system.copy_state(self.game)
        success = self._perform_action(action)
        if success:
            self.current_state = self._get_state()
        state_new = self.reward_system.copy_state(self.game)
        reward = self.reward_system.get_reward(state_old, action, state_new, success)
        info = self.get_info()
        done = self.get_done(info)
        return self.current_state, reward, done, info

    def get_done(self, info):
        game_done = self.game.context.done
        no_available_actions = len(info) == 0
        lost_customers = (self.game.context.customers <= 0 and not self.game.context.is_new_game)
        return game_done or no_available_actions or lost_customers

    def _perform_start_sprint_action(self) -> bool:
        can_start_sprint = self.game.backlog.can_start_sprint()
        if can_start_sprint:
            self.game.backlog_start_sprint()
        return can_start_sprint

    def _perform_decomposition(self) -> bool:
        is_release_available = self.game.userstories.release_available
        if is_release_available:
            self.game.userstories_start_release()
        return is_release_available
    
    def _perform_release(self) -> bool:
        is_release_available = self.game.hud.release_available
        if is_release_available:
            self.game.hud_release_product()
        return is_release_available

    def _perform_buy_robot(self) -> bool:
        room_num = self.game.get_min_not_full_room_number()
        if room_num == -1:
            return False
        worker_count_before = self.game.context.available_developers_count
        self.game.buy_robot(room_num)
        worker_count = self.game.context.available_developers_count
        return worker_count_before != worker_count
    
    def _perform_buy_room(self) -> bool:
        room_num = self.game.get_min_available_to_buy_room_number()
        if room_num == -1:
            return False
        worker_count_before = self.game.context.available_developers_count
        self.game.buy_room(room_num)
        worker_count = self.game.context.available_developers_count
        return worker_count_before != worker_count
    
    def _perform_statistical_research(self) -> bool:
        if not self.game.userstories.statistical_research_available:
            return False
        stories_before = len(self.game.userstories.stories_list)
        self.game.press_statistical_research()
        stories_after = len(self.game.userstories.stories_list)
        return stories_before != stories_after
    
    def _perform_user_survey(self) -> bool:
        if not self.game.userstories.user_survey_available:
            return False
        stories_before = len(self.game.userstories.stories_list)
        self.game.press_user_survey()
        stories_after = len(self.game.userstories.stories_list)
        return stories_before != stories_after
    
    def _perform_action(self, action: int) -> bool:
        # we'll assume that action in range(0, max_action_num)
        if action == START_SPRINT:
            return self._perform_start_sprint_action()
        if action == DECOMPOSE:
            return self._perform_decomposition()
        if action == RELEASE:
            return self._perform_release()
        if action == BUY_ROBOT:
            return self._perform_buy_robot()
        if action == BUY_ROOM:
            return self._perform_buy_room()
        if action == STATISTICAL_RESEARCH:
            return self._perform_statistical_research()
        if action == USER_SURVEY:
            return self._perform_user_survey()
        
        return self._perform_action_card(action - self.meta_action_dim)

    def _perform_action_card(self, action: int) -> bool:
        if action < self.userstory_env.max_action_num:
            return self._perform_action_userstory(action)
        
        card_id = action - self.userstory_env.max_action_num
        if card_id < self.backlog_env.backlog_max_action_num:
            return self._perform_action_backlog_card(card_id)
        
        card_id = card_id - self.backlog_env.backlog_max_action_num
        return self._perform_remove_sprint_card(card_id)

    def _perform_action_backlog_card(self, action: int) -> bool:
        card: Card = self.backlog_env.get_card(action)
        
        if card is None:
            return False
        
        hours_after_move = self.game.backlog.calculate_hours_sum() + card.info.hours
        if hours_after_move > self.game.backlog.get_max_hours():
            return False

        self.game.move_backlog_card(card)
        return True

    def _perform_action_userstory(self, action: int) -> bool:
        card = self.userstory_env.get_encoded_card(action)

        if card is None or not self.game.userstories.available:
            return False

        if not card.is_movable:
            return False

        self.game.move_userstory_card(card)
        return True

    def _perform_remove_sprint_card(self, card_id: int) -> bool:
        card = None
        backlog_env = self.backlog_env

        if card_id < backlog_env.sprint_commons_count:
            card = self._get_card(backlog_env.sprint_commons, card_id)

        bug_card_id = card_id - backlog_env.sprint_commons_count
        if card is None and bug_card_id < backlog_env.sprint_bugs_count:
            card = self._get_card(backlog_env.sprint_bugs, bug_card_id)

        tech_debt_card_id = bug_card_id - backlog_env.sprint_bugs_count
        if card is None and tech_debt_card_id < backlog_env.sprint_tech_debt_count:
            card = self._get_card(backlog_env.sprint_tech_debt, tech_debt_card_id)

        if card is None:
            return False
        self.game.move_sprint_card(card)
        return True

    def _get_card(self, sampled, index):
        if 0 <= index < len(sampled):
            return sampled[index]
        return None


class LoggingEnv(ProductOwnerEnv):
    def step(self, action: int):
        new_state, reward, done, info = super().step(action)
        print(action, reward)
        return new_state, reward, done, info

class BuggyProductOwnerEnv(ProductOwnerEnv):
    def __init__(self, userstory_env=None, backlog_env=None, with_info=True,
                 seed=None, card_picker_seed=None):
        super().__init__(userstory_env, backlog_env, with_info,
                         seed=seed, card_picker_seed=card_picker_seed)
        self.game = get_buggy_game_1(seed=seed)
        self.current_state = self._get_state()
    
    def reset(self, seed=None, card_picker_seed=None):
        self.game = get_buggy_game_1(seed=seed)
        super()._reset_card_picker_random_gen(card_picker_seed)
        self.current_state = self._get_state()
        return self.current_state
