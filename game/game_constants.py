from enum import Enum


UserCardType = Enum("UserCardType", ["S", "M", "L", "XL", "BUG", "TECH_DEBT"])


class GlobalConstants:
    developer_hours = 10
    statistical_research_cost = 80000
    MONEY_GOAL = 1000000
    MAX_WORKER_COUNT = 4
    NEW_WORKER_COST = 50000
    NEW_ROOM_COST = 200000
    NEW_ROOM_MULTIPLIER = 1.5
    user_survey_cost = 160000

    USERSTORY_LOYALTY = {UserCardType.S: [0.025, 0.08], UserCardType.M: [0.075, 0.175],
                         UserCardType.L: [0.125, 0.35], UserCardType.XL: [0.25, 0.5]}
    USERSTORY_CUSTOMER = {UserCardType.S: [1, 3.5], UserCardType.M: [2.5, 7],
                          UserCardType.L: [5, 14], UserCardType.XL: [10, 28]}

    AMOUNT_CREDIT_PAYMENT = 9000

    BLANK_SPRINT_LOYALTY_DECREMENT = {
        6: -0.05,
        9: -0.1,
        12: -0.15
    }
    min_key_blank_sprint_loyalty = min(BLANK_SPRINT_LOYALTY_DECREMENT.keys())

    BLANK_SPRINT_CUSTOMERS_DECREMENT = {
        6: -0.5,
        9: -1.0,
        12: -1.5
    }

    USERSTORY_FLOATING_PROFIT = {3: [1, 1.3], 6: [
        0.7, 0.9], 9: [0.2, 0.6], 12: [-0.2, 0.1]}
    sorted_keys_userstory_floating_profit = sorted(
        USERSTORY_FLOATING_PROFIT.keys())

    BUG_SPAM_PROBABILITY = 0.25
    TECH_DEBT_SPAWN_PROBABILITY = 0.5
