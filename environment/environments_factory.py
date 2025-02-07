from environment import CreditPayerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.reward_sytem import FullPotentialCreditRewardSystem
from pipeline.aggregator_study import update_reward_system_config


class EnvironmentFactory:
    def create_credit_env(self):
        userstory_env = UserstoryEnv(4, 0, 0)
        backlog_env = BacklogEnv(12, 0, 0, 0, 0, 0)
        reward_system = FullPotentialCreditRewardSystem(config={}, coefficient=1)

        env = CreditPayerEnv(
            userstory_env,
            backlog_env,
            with_end=True,
            with_info=True,
            reward_system=reward_system,
        )
        update_reward_system_config(env, reward_system)

        return env
