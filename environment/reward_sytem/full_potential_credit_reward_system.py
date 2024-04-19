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
        first = self.get_first_sum_part(state_new) - self.get_first_sum_part(state_old)
        # if self.get_sprint_hours(state_old) > 0 and first < 0:
        #     first = 0
        second = self.get_second_sum_part(state_new) - self.get_second_sum_part(state_old)
        third = self.get_third_sum_part(state_new) - self.get_third_sum_part(state_old)
        fourth = self.get_fourth_sum_part(state_new) - self.get_fourth_sum_part(state_old)

        reward = first + second + third + fourth

        return reward

    def get_first_sum_part(self, state) -> float:
        blank_sprint_counter = self.get_blank_sprint_counter(state)
        customers = self.get_customers(state)
        loyalty = self.get_loyalty(state)
        return max((7 - blank_sprint_counter) * customers * loyalty, 0) * self.potential_weight

    def get_second_sum_part(self, state) -> float:
        blank_counter = self.get_blank_sprint_counter(state)
        customers = self.get_customers(state)
        loyalty = self.get_loyalty(state)

        if blank_counter - 4 < 9:
            indexes = torch.arange(max(blank_counter - 4, 3), 9)
            sums = (3 + indexes) / 2 * (indexes - 2) / 3
            result = (customers - 0.5 * sums) * (loyalty - 0.05 * sums)
            return torch.sum(result).item() * self.potential_weight
        else:
            return 0

    def get_third_sum_part(self, state) -> float:
        customers = self.get_customers(state)
        loyalty = self.get_loyalty(state)

        n = (2 * customers - 11) // 3

        if n < 1:
            return 0
        indexes = torch.arange(1, n + 1)
        customers_part = (customers - 5.5 - 1.5 * indexes)
        loyalty_part = (loyalty - 0.55 - 0.15 * indexes)
        loyalty_part[loyalty_part < 0.8] = 0.8
        result = customers_part * loyalty_part
        return torch.sum(result).item() * self.potential_weight

    def get_fourth_sum_part(self, state):
        return self.get_money(state) * self.money_weight
