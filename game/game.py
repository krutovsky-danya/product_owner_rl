from game.backlog.backlog import Backlog
from game.userstories.userstories import UserStories
from game.hud.hud import HUD
from game.office.office import Offices
from game import game_global as Global
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from game.backlog_card.card_info import CardInfo
from game.userstory_card.bug_user_story_info import BugUserStoryInfo
from game.userstory_card.tech_debt_user_story_info import TechDebtInfo
import random


backlog = Backlog()
userstories = UserStories()
hud = HUD()
office = Offices()

sprint_cost = 0
completed_us = []
cards_in_sprint = []
is_first_release = True
force_td_spawn = False

class ProductOwnerGame:
    def __init__(self):
        self.backlog = Backlog()
        self.userstories = UserStories()
        self.hud = HUD()
        self.office = Offices()

        self.sprint_cost = 0
        self.completed_us = []
        self.cards_in_sprint = []
        self.is_first_release = True
        self.force_td_spawn = False
    
    def _on_backlog_start_sprint(self, cards_info):
        self.cards_in_sprint = cards_info
        self.hud.increase_progress(cards_in_sprint)
        self._next_sprint()


def load_game():  # !
    global backlog, userstories, hud, office, sprint_cost, \
        completed_us, cards_in_sprint, is_first_release, force_td_spawn
    Global.project_name = "MLTest"
    sprint_cost = 0
    completed_us = []
    cards_in_sprint = []
    is_first_release = True
    force_td_spawn = False
    Global.reload_game()

    backlog = Backlog()
    userstories = UserStories()
    hud = HUD()
    office = Offices()


def _on_backlog_start_sprint(cards_info):
    global cards_in_sprint
    cards_in_sprint = cards_info
    hud.increase_progress(cards_in_sprint)
    _next_sprint()


def backlog_start_sprint():  # !
    if backlog.can_start_sprint():
        cards = backlog.on_start_sprint_pressed()
        if cards:
            backlog.clear_sprint()
            _on_backlog_start_sprint(cards)


def _on_userstories_start_release(cards_info):
    for i in cards_info:
        us: UserStoryCardInfo = i
        us.is_decomposed = True
    backlog.generate_cards()


def userstories_start_release():  # !
    # todo добавить проверку на превышение доступного количества часов
    if userstories.release_available:
        cards = userstories.on_start_release_pressed()
        userstories.clear_release()
        _on_userstories_start_release(cards)


def _next_sprint():
    global cards_in_sprint
    Global.current_sprint += 1
    _update_tech_debt_impact()
    for i in cards_in_sprint:
        card: CardInfo = i
        us: UserStoryCardInfo = Global.current_stories[card.us_id]
        us.related_cards.remove(card)
        if len(us.related_cards) == 0:
            us.completed = True
            us.completed_part = 1
            completed_us.append(us)
    cards_in_sprint = []
    Global.set_money(Global.get_money() + _get_sprint_money())
    Global.current_sprint_hours = 0
    backlog.generate_cards()


def _update_tech_debt_impact():
    impact_count = 0
    for card in cards_in_sprint:
        if card.card_type != Global.UCType.TECH_DEBT:
            impact_count += 1
    full_tech_debt_debuff = 0
    for key in Global.current_tech_debt.keys():
        tech_debt_debuff = impact_count * Global.current_tech_debt[key].hours_debuff_increment
        Global.current_tech_debt[key].full_hours_debuff += tech_debt_debuff
        full_tech_debt_debuff += Global.current_tech_debt[key].full_hours_debuff

    _update_tech_debt_impact_stories(Global.current_stories, full_tech_debt_debuff)
    _update_tech_debt_impact_stories(Global.available_stories, full_tech_debt_debuff)


def _update_tech_debt_impact_stories(stories, full_tech_debt_debuff: int):
    for i in stories.values():
        us: UserStoryCardInfo = i
        if us.card_type == Global.UCType.TECH_DEBT:
            continue
        for card in us.related_cards:
            card.hours = card.base_hours + full_tech_debt_debuff


def _update_loyalty():
    if Global.is_new_game:
        return
    for i in Global.current_bugs.keys():
        bug: BugUserStoryInfo = Global.current_bugs[i]
        Global.set_loyalty(Global.get_loyalty() + bug.loyalty_debuff)
        bug.loyalty_debuff += bug.loyalty_increment
        Global.customers += bug.customers_debuff
        bug.customers_debuff += bug.customers_increment
    if Global.blank_sprint_counter >= Global.min_key_bs_lty:
        delta_loyalty = Global.interpolate(Global.blank_sprint_counter,
                                           Global.BLANK_SPRINT_LOYALTY_DECREMENT)
        Global.set_loyalty(Global.get_loyalty() + delta_loyalty)
        d_customers = Global.interpolate(Global.blank_sprint_counter,
                                         Global.BLANK_SPRINT_CUSTOMERS_DECREMENT)
        Global.customers += d_customers


def _get_credit_payment() -> int:
    if Global.credit > 0:
        credit_payment = min(Global.credit, Global.AMOUNT_CREDIT_PAYMENT)
        Global.credit -= credit_payment
        return credit_payment
    return 0


def _get_sprint_money():
    sprint_money = Global.customers * Global.get_loyalty() * 300
    if Global.current_sprint_hours > 0:
        sprint_money -= Global.available_developers_count * Global.worker_cost
    Global.blank_sprint_counter += 1
    _update_loyalty()
    sprint_money -= _get_credit_payment()
    return sprint_money


def _on_hud_release_product():
    global force_td_spawn
    _update_profit()
    force_td_spawn = not Global.is_first_bug
    _check_and_spawn_tech_debt()
    _check_and_spawn_bug()
    _update_tech_debt_impact()
    hud.release_available = False
    if Global.is_new_game:
        Global.is_new_game = False
        userstories.disable_restrictions()
        office.toggle_purchases(True)


def hud_release_product():  # !
    if hud.release_available:
        _on_hud_release_product()


def _update_profit():
    global is_first_release, completed_us
    Global.blank_sprint_counter = 0
    for i in completed_us:
        us: UserStoryCardInfo = i
        if is_first_release:
            Global.customers = 25
            Global.set_loyalty(4.0)
            is_first_release = False
        elif id(us) in Global.current_bugs:
            Global.current_bugs.pop(id(us))
        elif id(us) in Global.current_tech_debt:
            Global.current_tech_debt.pop(id(us))
        else:
            _update_profit_ordinary_card(us)
        Global.release_color(us.card_type, us.color)
        Global.current_stories.pop(id(us))
    completed_us = []


def _update_profit_ordinary_card(us: UserStoryCardInfo):
    sprints_spent = Global.current_sprint - us.spawn_sprint
    for us_fp_key in Global.sorted_keys_us_fp:
        us_fp = Global.US_FLOATING_PROFIT[us_fp_key]
        if sprints_spent <= us_fp_key or us_fp_key == Global.sorted_keys_us_fp[-1]:
            ran_usr_to_bring = us.customers_to_bring * random.uniform(us_fp[0], us_fp[1])
            Global.customers += Global.stepify(ran_usr_to_bring, 0.01)
            ran_lty_to_bring = us.loyalty * random.uniform(us_fp[0], us_fp[1])
            Global.set_loyalty(Global.get_loyalty() + Global.stepify(ran_lty_to_bring, 0.01))
            break


def _check_and_spawn_bug():
    if _is_ready_to_spawn_bug():
        bug_us = BugUserStoryInfo()
        userstories.add_us(bug_us)
        Global.current_bugs[id(bug_us)] = bug_us
        Global.is_first_bug = False


def _is_ready_to_spawn_bug():
    has_color_for_bug = len(Global.current_bugs) < 7
    paid_credit = Global.credit == 0
    chanced_bug = random.uniform(0, 1) < Global.BUG_SPAM_PROBABILITY
    has_td = len(Global.current_tech_debt) > 0
    return has_color_for_bug and paid_credit and (chanced_bug or has_td or Global.is_first_bug)


def _check_and_spawn_tech_debt():
    global force_td_spawn
    if _is_ready_to_spawn_tech_debt():
        force_td_spawn = False
        tech_debt = TechDebtInfo()
        userstories.add_us(tech_debt)
        Global.current_tech_debt[id(tech_debt)] = tech_debt
        Global.is_first_tech_debt = False


def _is_ready_to_spawn_tech_debt():
    has_color_for_td = len(Global.current_tech_debt) < 7
    paid_credit = Global.credit == 0
    chanced_td = random.uniform(0, 1) < Global.TECH_DEBT_SPAWN_PROBABILITY
    return has_color_for_td and paid_credit and chanced_td and force_td_spawn


def buy_robot(room_num):  # !
    # todo снять привязку к номерам комнат. Добавлять робота в комнату, где ещё есть место
    # (в комнату с минимальным номером из тех, что удовлетворяют этому условию)
    room = office.offices[Global.clamp(room_num, 0, len(office.offices) - 1)]
    has_bought = room.on_buy_robot_button_pressed()
    if has_bought:
        Global.buy_robot()


def buy_room(room_num):  # !
    # todo снять привязку к номерам комнат. Добавлять комнату при покупке
    room = office.offices[Global.clamp(room_num, 0, len(office.offices) - 1)]
    has_bought = room.on_buy_room_button_pressed()
    if has_bought:
        Global.buy_room()


def _on_userstory_card_dropped(card, is_on_left: bool):
    if is_on_left:
        userstories.on_stories_card_dropped(card)
    else:
        userstories.on_release_card_dropped(card)


def move_userstory_card(card_num):  # !
    # зависит от того, что мы будем с этим делать:
    # предполагается ли, что модель может закинуть карточку в декомпозицию,
    # а потом вытащить её от туда? то же самое с бэклогом
    if userstories.available:
        stories = userstories.stories_list
        if len(stories) > 0:
            card = stories[Global.clamp(card_num, 0, len(stories) - 1)]
            if card.is_movable:
                _on_userstory_card_dropped(card, False)


def move_backlog_card(card_num):  # !
    cards = backlog.backlog
    if len(cards) > 0:
        card = cards[Global.clamp(card_num, 0, len(cards) - 1)]
        if card.is_movable:
            backlog.backlog.remove(card)
            backlog.sprint.append(card)


def press_statistical_research():  # !
    if userstories.statistical_research_available:
        userstories.on_statistical_research_pressed()


def press_user_survey():  # !
    if userstories.user_survey_available:
        userstories.on_user_survey_pressed()
