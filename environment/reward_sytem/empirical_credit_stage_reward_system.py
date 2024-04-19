from .empirical_reward_system import EmpiricalRewardSystem

PURCHASE_ACTIONS = {3, 4, 5, 6}
LATE_PURCHASE_SPRINT = 29


class EmpiricalCreditStageRewardSystem(EmpiricalRewardSystem):
    def __init__(self, with_late_purchase_punishment, config: dict) -> None:
        super().__init__(config)
        self.with_late_purchase_punishment = with_late_purchase_punishment
        self.potential_weight = 3
        self.late_purchases_reward = -100

    def get_reward(self, state_old, action, state_new, success) -> float:
        reward = super().get_reward(state_old, action, state_new, success)
        reward += self.get_credit_payer_reward(state_old, state_new)
        if self.with_late_purchase_punishment:
            reward += self.get_late_purchases_punishment(state_old, action, state_new)
        return reward

    def get_credit_payer_reward(self, state_old, state_new) -> float:
        potential_old = self.get_potential(state_old)
        potential_new = self.get_potential(state_new)

        diff = potential_new - potential_old
        return diff

    def get_potential(self, state):
        loyalty = self.get_loyalty(state)
        customers = self.get_customers(state)

        potential = loyalty * customers
        return potential * self.potential_weight

    def get_late_purchases_punishment(self, state_old, action, state_new) -> float:
        money_diff = self.get_money(state_old) - self.get_money(state_new)
        if money_diff >= 0:
            return 0
        if self.get_sprint(state_new) < LATE_PURCHASE_SPRINT:
            return 0
        if action not in PURCHASE_ACTIONS:
            return 0
        return self.late_purchases_reward


class BoundedEmpiricalCreditStageRewardSystem(EmpiricalCreditStageRewardSystem):
    def __init__(self, with_late_purchase_punishment, config: dict) -> None:
        super().__init__(with_late_purchase_punishment, config)
        self.potential_weight = 0.03
        self.money_weight = 1e-5
        self.late_purchases_reward = -0.5
