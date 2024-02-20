from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from game.backlog_card.backlog_card import Card
from game.game import ProductOwnerGame
from game.game_constants import UserCardType, GlobalConstants
import torch
import numpy as np
from game.game_generators import get_buggy_game_1

BUG = UserCardType.BUG
TECH_DEBT = UserCardType.TECH_DEBT


class ProductOwnerEnv:
    def __init__(self, userstory_env=None, backlog_env: BacklogEnv = None, with_sprint=True):
        self.game = ProductOwnerGame()
        self.backlog_env = BacklogEnv() if backlog_env is None else backlog_env
        self.backlog_env.with_sprint = with_sprint
        self.userstory_env = UserstoryEnv() if userstory_env is None else userstory_env

        self.meta_space_dim = 18

        self.state_dim = self.meta_space_dim + \
            self.userstory_env.userstory_space_dim + \
            self.backlog_env.backlog_space_dim

        self.with_sprint = with_sprint

        if self.with_sprint:
            self.state_dim += self.backlog_env.sprint_space_dim
        
        self.current_state = self._get_state()

        self.meta_action_dim = 7
        self.userstory_max_action_num = + \
            self.userstory_env.us_common_count + \
            self.userstory_env.us_bug_count + \
            self.userstory_env.us_td_count
        self.backlog_max_action_num = + \
            self.backlog_env.backlog_commons_count + \
            self.backlog_env.backlog_bugs_count + \
            self.backlog_env.backlog_tech_debt_count

        self.sprint_max_action_num = 0
        if self.with_sprint:
            self.sprint_max_action_num = + \
                self.backlog_env.sprint_commons_count + \
                self.backlog_env.sprint_bugs_count + \
                self.backlog_env.sprint_tech_debt_count
        self.action_n = self.meta_action_dim + \
            self.userstory_max_action_num + \
            self.backlog_max_action_num + \
            self.sprint_max_action_num

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
            self.game.backlog.calculate_hours_sum(),
            context.blank_sprint_counter,
            self.game.backlog.can_start_sprint(),
            self.game.hud.release_available,
            self.game.userstories.release_available,
            self.game.userstories.statistical_research_available,
            self.game.userstories.user_survey_available,
            *self._get_completed_cards_count(),
            *self.userstory_env.encode(self.game.userstories.stories_list),
            *self.backlog_env.encode(self.game.backlog)
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

    def step(self, action: int):
        # new_state, reward, done, info
        reward = 0
        credit_before = self.game.context.credit
        reward_bit = self._perform_action(action)
        credit_after = self.game.context.credit
        if credit_before > 0 and credit_after <= 0:
            reward += 10
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
                reward_for_endgame = -50
        else:
            reward_for_endgame = 0
        return reward_for_endgame

    def _perform_start_sprint_action(self) -> int:
        if not self.game.backlog.can_start_sprint():
            return -10
        self.game.backlog_start_sprint()
        return 1

    def _perform_decomposition(self) -> int:
        is_release_available = self.game.userstories.release_available
        if is_release_available:
            self.game.userstories_start_release()
            return 1
        return -10
    
    def _perform_release(self) -> int:
        is_release_available = self.game.hud.release_available
        if is_release_available:
            self.game.hud_release_product()
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
    
    def _perform_statistical_research(self) -> int:
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
            return self._perform_statistical_research()
        if action == 6:
            return self._perform_user_survey()
        
        return self._perform_action_card(action - 7)

    def _perform_action_card(self, action: int) -> int:
        if action < self.userstory_max_action_num:
            return self._perform_action_userstory(action)
        
        card_id = action - self.userstory_max_action_num
        if card_id < self.backlog_max_action_num:
            return self._perform_action_backlog_card(card_id)
        
        card_id = card_id - self.backlog_max_action_num
        return self._perform_remove_sprint_card(card_id)

    def _perform_action_backlog_card(self, action: int) -> int:
        card: Card = None
        backlog_env = self.backlog_env

        if action < backlog_env.backlog_commons_count:
            card = self._get_card(backlog_env.backlog_commons, action)

        bug_card_id = action - backlog_env.backlog_commons_count
        if card is None and bug_card_id < backlog_env.backlog_bugs_count:
            card = self._get_card(backlog_env.backlog_bugs, bug_card_id)

        tech_debt_card_id = bug_card_id - backlog_env.backlog_bugs_count
        if card is None and tech_debt_card_id < backlog_env.backlog_tech_debt_count:    
            card = self._get_card(backlog_env.backlog_tech_debt, tech_debt_card_id)
        
        if card is None:
            return -10
        
        hours_after_move = self.game.backlog.calculate_hours_sum() + card.info.hours
        if hours_after_move > self.game.backlog.get_max_hours():
            return -1

        self.game.move_backlog_card(card)
        return 1

    def _perform_action_userstory(self, action: int) -> int:
        if action < self.userstory_env.us_common_count:
            card = self._get_card(self.userstory_env.userstories_common, action)
        elif action - self.userstory_env.us_common_count < self.userstory_env.us_bug_count:
            card = self._get_card(
                self.userstory_env.userstories_bugs, action - self.userstory_env.us_common_count)
        else:
            card = self._get_card(
                self.userstory_env.userstories_td,
                action - self.userstory_env.us_common_count - self.userstory_env.us_bug_count)

        if card is None or not self.game.userstories.available:
            return -10

        if not card.is_movable:
            return -1

        self.game.move_userstory_card(card)
        return 1

    def _perform_remove_sprint_card(self, card_id: int) -> int:
        card = None
        backlog_env = self.backlog_env

        if card_id < backlog_env.sprint_commons_count:
            card = self._get_card(backlog_env.sprint_commons, card_id)

        bug_card_id = card_id - backlog_env.sprint_commons_count
        if card is None and bug_card_id < backlog_env.sprint_bugs_count:
            card = self._get_card(backlog_env.sprint_bugs, bug_card_id)

        texh_debt_card_id = bug_card_id - backlog_env.sprint_bugs_count
        if card is None and texh_debt_card_id < backlog_env.sprint_tech_debt_count:
            card = self._get_card(backlog_env.sprint_tech_debt, texh_debt_card_id)

        if card is None:
            return -10
        self.game.move_sprint_card(card)
        return -2

    def _get_card(self, sampled, index):
        if 0 <= index < len(sampled):
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

class LoggingEnv(ProductOwnerEnv):
    def step(self, action: int):
        new_state, reward, done, info = super().step(action)
        print(action, reward)
        return new_state, reward, done, info

class BuggyProductOwnerEnv(ProductOwnerEnv):
    def __init__(self, userstory_env=None, backlog_env=None):
        super().__init__(userstory_env, backlog_env)
        self.game = get_buggy_game_1()
        self.current_state = self._get_state()
    
    def reset(self):
        self.game = get_buggy_game_1()
        self.current_state = self._get_state()
        return self.current_state
