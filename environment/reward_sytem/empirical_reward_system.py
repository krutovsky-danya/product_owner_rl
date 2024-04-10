from .base_reward_system import BaseRewardSystem

START_SPRINT = 0

class EmpiricalRewardSystem(BaseRewardSystem):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.wrong_action_reward = -10
        self.credit_payment_reward = 10
        self.win_reward = 500
        self.lose_reward = -50
        self.remove_sprint_card_reward = -2
        self.valid_action_reward = 1

    def get_reward(self, state_old, action, state_new) -> float:
        if (state_old == state_new).all():
            return self.wrong_action_reward
        reward = 0
        if self.get_credit(state_old) > 0 and self.get_credit(state_new) <= 0:
            reward += self.credit_payment_reward
        reward += self.get_done_reward(state_new)
        reward += self.get_action_reward(state_old, action, state_new)

        return reward

    def get_done_reward(self, state_new) -> float:
        done = self.get_done(state_new)
        if not done:
            return 0
        if self.get_money(state_new) > 10:
            return self.win_reward
        return self.lose_reward

    def get_action_reward(self, state_old, action, state_new) -> float:
        if action == START_SPRINT:
            return self.get_reward_for_starting_sprint(state_old, state_new)
        if action in self.config["remove_sprint_card_actions"]:
            return self.remove_sprint_card_reward
        return self.valid_action_reward

    def get_reward_for_starting_sprint(self, state_old, state_new) -> float:
        money_before = self.get_money(state_old)
        money_after = self.get_money(state_new)
        base_reward = money_after - money_before
        if base_reward < 0:
            return base_reward
        if self.get_sprint_hours(state_old) > 0:
            return base_reward / 10
        return 0


class BoundedEmpiricalRewardSystem(EmpiricalRewardSystem):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.wrong_action_reward = -0.1
        self.credit_payment_reward = 0.1
        self.win_reward = 1
        self.lose_reward = -1
        self.remove_sprint_card_reward = -0.02
        self.valid_action_reward = 0.01
