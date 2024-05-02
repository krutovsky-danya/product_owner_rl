from environment.reward_sytem import EmpiricalRewardSystem
import torch


class FullPotentialCreditRewardSystem(EmpiricalRewardSystem):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.remove_sprint_card_reward = 0
        self.valid_action_reward = 0
        self.get_action_reward = lambda old, action, new: 0
        self.potential_weight = 0.3
        self.money_weight = 1e-3

    def get_reward(self, state_old, action, state_new, success) -> float:
        reward = super().get_reward(state_old, action, state_new, success)
        reward += self.get_credit_payer_reward(state_old, state_new)

        return reward

    def get_credit_payer_reward(self, state_old, state_new) -> float:
        potential_income_old = self.get_potential_income(state_old)
        potential_income_new = self.get_potential_income(state_new)

        reward = potential_income_new - potential_income_old

        return reward

    def get_potential_income(self, state) -> float:
        return self.get_income_before_favor_loss(state) + \
               self.get_income_favor_loss_start(state) + \
               self.get_income_favor_loss_end(state) + \
               self.get_weighted_money(state)

    def get_income_before_favor_loss(self, state) -> float:
        blank_sprint_counter = self.get_blank_sprint_counter(state)
        sprints_before_loss = max(7 - blank_sprint_counter, 0)
        customers = self.get_customers(state)
        loyalty = self.get_loyalty(state)
        weighted_potential_income = customers * loyalty * self.potential_weight
        return sprints_before_loss * weighted_potential_income

    def get_income_favor_loss_start(self, state) -> float:
        blank_counter = self.get_blank_sprint_counter(state)
        customers = self.get_customers(state)
        loyalty = self.get_loyalty(state)

        if blank_counter >= 13:
            return 0
        indexes = torch.arange(max(blank_counter - 4, 3), 9)
        # сумма арифметической прогрессии с разностью 1, первым элементом 3 и
        # последним, входящим в сумму, элементом n равна
        # <среднее между первым и последним элементами суммы> * <число элементов в сумме> =
        # = (3 + n) / 2 * (n - 3 + 1)
        sums = (3 + indexes) / 2 * (indexes - 2)
        # данные значения взяты из заданного в game уменьшения лояльности и пользователей
        # (game_constants: BLANK_SPRINT_LOYALTY_DECREMENT и BLANK_SPRINT_CUSTOMERS_DECREMENT +
        # интерполяция)
        customers_loss = -0.5 * sums / 3
        loyalty_loss = -0.05 * sums / 3
        potential_incomes_per_sprint = (customers - customers_loss) * (loyalty - loyalty_loss)
        potential_income = torch.sum(potential_incomes_per_sprint).item()
        return potential_income * self.potential_weight

    def get_income_favor_loss_end(self, state) -> float:
        customers = self.get_customers(state)
        loyalty = self.get_loyalty(state)

        # на момент, когда число убывающих за спринт пользователей становится стабильным
        # и равным 1.5, количество уже ушедших пользователей равно 5.5
        # после потери всех пользователей агент больше не будет получать прибыль,
        # так что сумму имеет смысл считать только до этого момента
        # то есть в сумме будет n слагаемых, где при n выполняется неравенство
        # customers - 5.5 - n * 1.5 < 0
        # n * 1.5 > customers - 5.5
        # n > (customers - 5.5) / 1.5 = (2 * customers - 11) / 3
        n = (2 * customers - 11) // 3

        if n < 1:
            return 0
        indexes = torch.arange(1, n + 1)
        customers_part = (customers - 5.5 - 1.5 * indexes)
        loyalty_part = (loyalty - 0.55 - 0.15 * indexes)
        # лояльность не может упасть ниже 0.8
        loyalty_part[loyalty_part < 0.8] = 0.8
        potential_incomes_per_sprint = customers_part * loyalty_part
        potential_income = torch.sum(potential_incomes_per_sprint).item()
        return potential_income * self.potential_weight

    def get_weighted_money(self, state):
        return self.get_money(state) * self.money_weight
