from enum import Enum
from copy import copy
import random


def clamp(x, minimum, maximum):
    if x < minimum:
        return minimum
    elif x > maximum:
        return maximum
    return x


def stepify(s, step):
    return (s // step) * step


UCType = Enum("UCType", ["S", "M", "L", "XL", "BUG", "TECH_DEBT"])
UserCardColor = Enum("UserCardColor",
                     ["BLUE", "GREEN", "ORANGE", "PINK", "PURPLE", "RED", "YELLOW"])
colors_for_use = [UserCardColor.BLUE, UserCardColor.GREEN, UserCardColor.ORANGE,
                  UserCardColor.PINK, UserCardColor.PURPLE, UserCardColor.RED,
                  UserCardColor.YELLOW]


project_name = ""

developer_hours = 10
available_developers_count = 2
worker_cost = 10000
_money = 200000
credit = 300000
_loyalty = 0
customers = 0
current_sprint = 1
current_sprint_hours = 0
current_rooms_counter = 1
available_stories = {}
current_stories = {}
current_bugs = {}
current_tech_debt = {}  # todo (не перезадается в годоте в reload_game)
is_first_bug = True  # не перезадается в годоте в reload_game, но это не так критично
is_first_tech_debt = True  # не перезадается в годоте в reload_game, но это не так критично

used_colors = {UCType.S: [], UCType.M: [], UCType.L: [],
               UCType.XL: [], UCType.BUG: [], UCType.TECH_DEBT: []}

MONEY_GOAL = 1000000
AMOUNT_CREDIT_PAYMENT = 9000

blank_sprint_counter = 0  # не перезадается в годоте в reload_game, но это не так критично

BLANK_SPRINT_LOYALTY_DECREMENT = {
    6: -0.05,
    9: -0.1,
    12: -0.15
}
min_key_bs_lty = min(BLANK_SPRINT_LOYALTY_DECREMENT.keys())

BLANK_SPRINT_CUSTOMERS_DECREMENT = {
    6: -0.5,
    9: -1.0,
    12: -1.5
}

US_LTY = {UCType.S: [0.025, 0.08], UCType.M: [0.075, 0.175],
          UCType.L: [0.125, 0.35], UCType.XL: [0.25, 0.5]}
US_USR = {UCType.S: [1, 3.5], UCType.M: [2.5, 7],
          UCType.L: [5, 14], UCType.XL: [10, 28]}
US_FLOATING_PROFIT = {3: [1, 1.3], 6: [0.7, 0.9], 9: [0.2, 0.6], 12: [-0.2, 0.1]}
sorted_keys_us_fp = sorted(US_FLOATING_PROFIT.keys())

statistical_research_cost = 80000
user_survey_cost = 160000
MAX_WORKER_COUNT = 4
NEW_WORKER_COST = 50000
NEW_ROOM_COST = 200000
NEW_ROOM_MULTIPLIER = 1.5
current_room_multiplier = 1

BUG_SPAM_PROBABILITY = 0.25
TECH_DEBT_SPAWN_PROBABILITY = 0.5

is_new_game = True
# use_new_year_theme
done = False


def reload_game():
    global _loyalty, customers, available_developers_count, _money, credit, current_sprint,\
        current_rooms_counter, current_room_multiplier, current_sprint_hours, available_stories, \
        current_stories, current_bugs, current_tech_debt, is_first_bug, is_first_tech_debt, \
        is_new_game, used_colors, blank_sprint_counter, done
    _loyalty = 0
    customers = 0
    available_developers_count = 2
    _money = 200000
    credit = 300000
    current_sprint = 1
    current_rooms_counter = 1
    current_room_multiplier = 1
    current_sprint_hours = 0
    available_stories = {}
    current_stories = {}
    current_bugs = {}
    current_tech_debt = {}
    is_first_bug = True
    is_first_tech_debt = True
    blank_sprint_counter = 0
    is_new_game = True
    done = False
    for i in used_colors.keys():
        used_colors[i] = []


def set_money(count):
    global _money
    _money = count
    sg_money_changed(_money)


def get_money():
    return _money


def set_loyalty(value):
    global _loyalty
    _loyalty = clamp(value, 0.8, 5)


def get_loyalty():
    return _loyalty


def buy_robot():
    global _money, available_developers_count
    _money -= NEW_WORKER_COST
    available_developers_count += 1
    sg_money_changed(_money)


def buy_room():
    global _money, current_room_multiplier, NEW_ROOM_COST, NEW_ROOM_MULTIPLIER, \
        current_rooms_counter, available_developers_count
    _money -= NEW_ROOM_COST * current_room_multiplier
    current_room_multiplier *= NEW_ROOM_MULTIPLIER
    current_rooms_counter += 1
    available_developers_count += 1
    sg_money_changed(_money)


def has_enough_money(need_money: int) -> bool:
    global _money
    return _money >= need_money


def get_unused_color(uc_type: UCType):
    global used_colors
    if len(used_colors[uc_type]) == 7:
        print("Не осталось не использованных цветов.")
        return
    # todo в Godot'е и python используются разные генераторы (псевдо-)случайных чисел
    cfu = copy(colors_for_use)
    for i in used_colors[uc_type]:
        cfu.remove(i)
    i = random.randint(0, len(cfu) - 1)
    color = cfu[i]
    used_colors[uc_type].append(color)
    return color


def release_color(us_type: UCType, color: UserCardColor):
    global used_colors
    used_colors[us_type].remove(color)


# def _is_new_year(): так как здесь нет графики, этот метод не нужен


def sg_money_changed(value):
    # print("sg_money_changed")
    _on_money_changed(value)


def _on_money_changed(money_val):
    if money_val < 0:
        game_over(False)
    elif money_val >= MONEY_GOAL:
        game_over(True)


def game_over(win: bool):
    global done
    done = True
    if win:
        print("win")
        save_to_leaderboard()
    else:
        print("loose")


def save_to_leaderboard():
    # отличается от годота
    print(f"score: {1000000 - current_sprint}")


def interpolate(value, table: dict):
    keys = sorted(table.keys())
    first_key = keys[0]
    if value <= first_key:
        return table[first_key]

    last_key = keys[-1]
    if value >= last_key:
        return table[last_key]

    for i in range(1, len(keys)):
        if keys[i - 1] < value <= keys[i]:
            a = table[keys[i - 1]]
            b = table[keys[i]]
            u = a + (value - keys[i - 1]) * (b - a) / (keys[i] - keys[i - 1])
            return u

    return None


# class CardColor
# графика, текстурки


if __name__ == "__main__":
    # print(get_unused_color(UCType(2)))
    # print(get_unused_color(UCType(2)))
    # print(get_unused_color(UCType(2)))
    # print(get_unused_color(UCType(2)))
    # print(get_unused_color(UCType(2)))
    # print(get_unused_color(UCType(2)))
    # print(get_unused_color(UCType(2)))
    # # print(UCType["S"])
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    # print(release_color(UCType(2), used_colors[UCType(2)][0]))
    print(12)

    d = {"d": 5, "f": 12}
    print("h" in d)
    print("d" in d)
    print(len(d))
    d["l"] = 17
    print(d["l"])

    print(min(BLANK_SPRINT_LOYALTY_DECREMENT.keys()))
