from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.environment import ProductOwnerEnv


USUAL_CREDIT_ENV_END_SPRINT = 35
EARLY_CREDIT_ENV_END_SPRINT = 32
LATE_PURCHASE_SPRINT = 29
PURCHASE_ACTIONS = {3, 4, 5, 6}


class CreditPayerEnv(ProductOwnerEnv):
    def __init__(self, userstory_env=None, backlog_env=None, with_end=False,
                 with_late_purchases_punishment=False, with_info=True):
        if userstory_env is None:
            userstory_env = UserstoryEnv(6, 0, 0)
        if backlog_env is None:
            backlog_env = BacklogEnv(12, 0, 0, 0, 0, 0)
        super().__init__(userstory_env, backlog_env, with_info)
        self.with_end = with_end
        self.with_late_purchases_punishment = with_late_purchases_punishment
        if self.with_end:
            self.end_sprint = USUAL_CREDIT_ENV_END_SPRINT
        else:
            self.end_sprint = EARLY_CREDIT_ENV_END_SPRINT

    def step(self, action: int):
        context = self.game.context
        loyalty_before = context.get_loyalty()
        customers_before = context.customers
        money_before = context.get_money()

        new_state, reward, done, info = super().step(action)

        done = self.game.context.current_sprint == self.end_sprint or done
        reward += self._get_credit_payer_reward(loyalty_before, customers_before)
        if self.with_end and self.with_late_purchases_punishment:
            reward += self._get_late_purchases_punishment(action, money_before)

        return new_state, reward, done, info

    def _get_credit_payer_reward(self, loyalty_before, customers_before):
        context = self.game.context
        loyalty_after = context.get_loyalty()
        customers_after = context.customers

        potential_before = loyalty_before * customers_before
        potential_after = loyalty_after * customers_after
        difference = potential_after - potential_before

        # formula for income money per sprint in the game:
        # income = customers * loyalty * 300
        reward = difference * 3

        return reward

    def _get_late_purchases_punishment(self, action, money_before):
        current_sprint = self.game.context.current_sprint
        difference = self.game.context.get_money() - money_before

        if action in PURCHASE_ACTIONS and current_sprint > LATE_PURCHASE_SPRINT and difference < 0:
            return -100
        return 0
