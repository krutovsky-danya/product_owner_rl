from game.game import ProductOwnerGame


class BaseRewardSystem:
    def __init__(self, config: dict) -> None:
        self.sprint_index = "sprint"
        self.money_index = "money"
        self.customers_index = "customers"
        self.loyalty_index = "loyalty"
        self.credit_index = "credit"
        self.sprint_hours_index = "sprint_hours"
        self.done_index = "done"
        self.blank_sprint_counter_index = "blank_sprint_counter"
        self.config = config

    def copy_state(self, game: ProductOwnerGame):
        state = {
            self.sprint_index: game.context.current_sprint,
            self.money_index: game.context.get_money(),
            self.customers_index: game.context.customers,
            self.loyalty_index: game.context.get_loyalty(),
            self.credit_index: game.context.credit,
            self.sprint_hours_index: game.backlog.calculate_hours_sum(),
            self.done_index: game.context.done,
            self.blank_sprint_counter_index: game.context.blank_sprint_counter
        }

        return state

    def get_reward(self, state_old, action, state_new, success) -> float:
        return 0

    def get_sprint(self, state) -> float:
        return state[self.sprint_index]

    def get_money(self, state) -> float:
        return state[self.money_index]

    def get_customers(self, state) -> float:
        return state[self.customers_index]

    def get_loyalty(self, state) -> float:
        return state[self.loyalty_index]

    def get_credit(self, state) -> float:
        return state[self.credit_index]

    def get_sprint_hours(self, state) -> float:
        return state[self.sprint_hours_index]

    def get_done(self, state) -> bool:
        done_int = state[self.done_index]
        return bool(done_int)

    def get_blank_sprint_counter(self, state) -> int:
        return int(state[self.blank_sprint_counter_index])
