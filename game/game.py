from game.backlog.backlog import Backlog
from game.backlog_card.backlog_card import Card
from game.userstories.userstories import UserStories
from game.hud.hud import HUD
from game.rooms.devroom.room import OfficeRoom
from game.office.office import Offices
from game.game_constants import UserCardType, GlobalConstants
from game.game_variables import GlobalContext
from game.backlog_card.card_info import CardInfo
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from game.userstory_card.userstory_card import UserStoryCard
from game.common_methods import interpolate, stepify, clamp
import random
from typing import List, Dict


class ProductOwnerGame:
    def __init__(self):
        self.context = GlobalContext()
        self.backlog = Backlog(self.context)
        self.userstories = UserStories(self.context)
        self.hud = HUD(self.context)
        self.office = Offices(self.context)

        self.sprint_cost = 0
        self.completed_us: List[UserStoryCardInfo] = []
        self.cards_in_sprint: List[CardInfo] = []
        self.is_first_release = True
        self.force_td_spawn = False

    def _on_backlog_start_sprint(self, cards_info):
        self.cards_in_sprint = cards_info
        self.hud.increase_progress(self.cards_in_sprint)
        self._next_sprint()

    def backlog_start_sprint(self):
        if self.backlog.can_start_sprint():
            cards = self.backlog.on_start_sprint_pressed()
            if cards is not None:
                self.backlog.clear_sprint()
                self._on_backlog_start_sprint(cards)

    def _on_userstories_start_release(self, cards_info):
        for i in cards_info:
            us: UserStoryCardInfo = i
            us.is_decomposed = True
        self.backlog.generate_cards()

    def userstories_start_release(self):  # !
        # todo добавить проверку на превышение доступного количества часов
        if self.userstories.release_available:
            cards = self.userstories.on_start_release_pressed()
            self.userstories.clear_release()
            self._on_userstories_start_release(cards)

    def _next_sprint(self):
        self.context.current_sprint += 1
        self._update_tech_debt_impact()
        for card in self.cards_in_sprint:
            us: UserStoryCardInfo = self.context.current_stories[card.us_id]
            us.related_cards.remove(card)
            if len(us.related_cards) == 0:
                us.completed = True
                us.completed_part = 1
                self.completed_us.append(us)
        self.cards_in_sprint = []
        current_money = self.context.get_money()
        money_to_set = current_money + self._get_sprint_money()
        self.context.set_money(money_to_set)
        self.context.current_sprint_hours = 0
        self.backlog.generate_cards()

    def _update_tech_debt_impact(self):
        impact_count = 0
        for card in self.cards_in_sprint:
            if card.card_type != UserCardType.TECH_DEBT:
                impact_count += 1
        full_tech_debt_debuff = 0
        current_tech_debt = self.context.current_tech_debt
        for item in current_tech_debt.values():
            tech_debt_debuff = impact_count * item.hours_debuff_increment
            item.full_hours_debuff += tech_debt_debuff
            full_tech_debt_debuff += item.full_hours_debuff

        self._update_tech_debt_impact_stories(
            self.context.current_stories, full_tech_debt_debuff)
        self._update_tech_debt_impact_stories(
            self.context.available_stories, full_tech_debt_debuff)

    def _update_tech_debt_impact_stories(self, stories: Dict[int, UserStoryCardInfo], full_tech_debt_debuff: int):
        for us in stories.values():
            if us.card_type == UserCardType.TECH_DEBT:
                continue
            for card in us.related_cards:
                card.hours = card.base_hours + full_tech_debt_debuff

    def _update_loyalty(self):
        if self.context.is_new_game:
            return
        for bug in self.context.current_bugs.values():
            current_loyalty = self.context.get_loyalty()
            loyalty_to_set = current_loyalty + bug.loyalty_debuff
            self.context.set_loyalty(loyalty_to_set)
            bug.loyalty_debuff += bug.loyalty_increment
            self.context.customers += bug.customers_debuff
            bug.customers_debuff += bug.customers_increment

        blank_sprint_counter = self.context.blank_sprint_counter
        if blank_sprint_counter >= GlobalConstants.min_key_blank_sprint_loyalty:
            delta_loyalty = interpolate(blank_sprint_counter,
                                        GlobalConstants.BLANK_SPRINT_LOYALTY_DECREMENT)
            self.context.set_loyalty(
                self.context.get_loyalty() + delta_loyalty)
            delta_customers = interpolate(blank_sprint_counter,
                                          GlobalConstants.BLANK_SPRINT_CUSTOMERS_DECREMENT)
            self.context.customers += delta_customers

    def _get_credit_payment(self) -> int:
        if self.context.credit > 0:
            credit_payment = min(self.context.credit,
                                 GlobalConstants.AMOUNT_CREDIT_PAYMENT)
            self.context.credit -= credit_payment
            return credit_payment
        return 0

    def _get_sprint_money(self):
        sprint_money = self.context.customers * self.context.get_loyalty() * 300
        if self.context.current_sprint_hours > 0:
            sprint_money -= self.context.available_developers_count * self.context.worker_cost
        self.context.blank_sprint_counter += 1
        self._update_loyalty()
        sprint_money -= self._get_credit_payment()
        return sprint_money

    def _on_hud_release_product(self):
        self._update_profit()
        self.force_td_spawn = not self.context.is_first_bug
        self._check_and_spawn_tech_debt()
        self._check_and_spawn_bug()
        self._update_tech_debt_impact()
        self.hud.release_available = False
        if self.context.is_new_game:
            self.context.is_new_game = False
            self.userstories.disable_restrictions()
            self.office.toggle_purchases(True)

    def hud_release_product(self):  # !
        if self.hud.release_available:
            self._on_hud_release_product()

    def _update_profit(self):
        self.context.blank_sprint_counter = 0
        for us in self.completed_us:
            if self.is_first_release:
                self.context.customers = 25
                self.context.set_loyalty(4.0)
                self.is_first_release = False
            elif id(us) in self.context.current_bugs:
                self.context.current_bugs.pop(id(us))
            elif id(us) in self.context.current_tech_debt:
                self.context.current_tech_debt.pop(id(us))
            else:
                self._update_profit_ordinary_card(us)
            self.context.color_storage.release_color(us.card_type, us.color)
            self.context.current_stories.pop(id(us))
        self.completed_us = []

    def _update_profit_ordinary_card(self, us: UserStoryCardInfo):
        sprints_spent = self.context.current_sprint - us.spawn_sprint
        for us_fp_key in GlobalConstants.sorted_keys_userstory_floating_profit:
            us_fp = GlobalConstants.USERSTORY_FLOATING_PROFIT[us_fp_key]
            if sprints_spent <= us_fp_key or us_fp_key == GlobalConstants.sorted_keys_userstory_floating_profit[-1]:
                ran_usr_to_bring = us.customers_to_bring * \
                    random.uniform(us_fp[0], us_fp[1])
                self.context.customers += stepify(ran_usr_to_bring, 0.01)
                ran_lty_to_bring = us.loyalty * \
                    random.uniform(us_fp[0], us_fp[1])
                self.context.set_loyalty(
                    self.context.get_loyalty() + stepify(ran_lty_to_bring, 0.01))
                break

    def _check_and_spawn_bug(self):
        if self._is_ready_to_spawn_bug():
            bug_us = BugUserStoryInfo(self.context.current_sprint, self.context.color_storage)
            self.userstories.add_us(bug_us)
            self.context.current_bugs[id(bug_us)] = bug_us
            self.context.is_first_bug = False

    def _is_ready_to_spawn_bug(self):
        has_color_for_bug = len(self.context.current_bugs) < 7
        paid_credit = self.context.credit == 0
        chanced_bug = random.uniform(
            0, 1) < GlobalConstants.BUG_SPAM_PROBABILITY
        has_td = len(self.context.current_tech_debt) > 0
        return has_color_for_bug and paid_credit and (chanced_bug or has_td or self.context.is_first_bug)

    def _check_and_spawn_tech_debt(self):
        if self._is_ready_to_spawn_tech_debt():
            self.force_td_spawn = False
            tech_debt = TechDebtInfo(self.context.current_sprint, self.context.color_storage)
            self.userstories.add_us(tech_debt)
            self.context.current_tech_debt[id(tech_debt)] = tech_debt
            self.context.is_first_tech_debt = False

    def _is_ready_to_spawn_tech_debt(self):
        has_color_for_td = len(self.context.current_tech_debt) < 7
        paid_credit = self.context.credit == 0
        chanced_td = random.uniform(
            0, 1) < GlobalConstants.TECH_DEBT_SPAWN_PROBABILITY
        return has_color_for_td and paid_credit and chanced_td and self.force_td_spawn

    def buy_robot(self, room_num):  # !
        room: OfficeRoom = self.office.offices[clamp(
            room_num, 0, len(self.office.offices) - 1)]
        has_bought = room.on_buy_robot_button_pressed()
        if has_bought:
            self.context.buy_robot()

    def buy_room(self, room_num):  # !
        room = self.office.offices[clamp(
            room_num, 0, len(self.office.offices) - 1)]
        has_bought = room.on_buy_room_button_pressed()
        if has_bought:
            self.context.buy_room()

    def _on_userstory_card_dropped(self, card: UserStoryCard, is_on_left: bool):
        if is_on_left and card.is_in_release:
            self.userstories.on_stories_card_dropped(card)
            card.is_in_release = False
        elif not is_on_left and not card.is_in_release:
            self.userstories.on_release_card_dropped(card)
            card.is_in_release = True

    def move_userstory_card(self, card):  # !
        if self.userstories.available:
            if isinstance(card, int):
                stories = self.userstories.stories_list
                if len(stories) > 0:
                    card = stories[clamp(card, 0, len(stories) - 1)]
                    if card.is_movable:
                        self._on_userstory_card_dropped(card, False)
            elif card is not None:
                if card.is_movable:
                    self._on_userstory_card_dropped(card, False)

    def move_backlog_card(self, card):  # !
        cards = self.backlog.backlog
        if len(cards) > 0:
            if isinstance(card, int):
                card = cards[clamp(card, 0, len(cards) - 1)]
                if card.is_movable and not card.is_in_sprint:
                    self.backlog.backlog.remove(card)
                    self.backlog.sprint.append(card)
                    card.is_in_sprint = True
            elif card is not None:
                if card.is_movable and not card.is_in_sprint:
                    self.backlog.backlog.remove(card)
                    self.backlog.sprint.append(card)
                    card.is_in_sprint = True

    def move_sprint_card(self, card: Card):
        cards = self.backlog.sprint
        if len(cards) == 0:
            return
        if card.is_movable and card.is_in_sprint:
            self.backlog.sprint.remove(card)
            self.backlog.backlog.append(card)
            card.is_in_sprint = False

    def press_statistical_research(self):  # !
        if self.userstories.statistical_research_available:
            self.userstories.on_statistical_research_pressed()

    def press_user_survey(self):  # !
        if self.userstories.user_survey_available:
            self.userstories.on_user_survey_pressed()
