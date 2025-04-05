from environment import CreditPayerEnv, ProductOwnerEnv
from environment.backlog_env import BacklogEnv
from environment.userstory_env import UserstoryEnv
from environment.reward_sytem import (
    FullPotentialCreditRewardSystem,
    StaircaseRewardSystem,
)
from pipeline.aggregator_study import update_reward_system_config


class EnvironmentFactory:
    def __init__(self, staircase_edge_sprint=100):
        self.staircase_edge_sprint = staircase_edge_sprint
        pass

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

    def create_full_env(self):
        userstory_env = UserstoryEnv(4, 2, 2)
        backlog_env = BacklogEnv(12, 6, 6, 0, 0, 0)
        reward_system = FullPotentialCreditRewardSystem(config={}, coefficient=1)

        env = ProductOwnerEnv(
            userstory_env,
            backlog_env,
            with_info=True,
            reward_system=reward_system,
        )
        update_reward_system_config(env, reward_system)

        return env

    def create_staircase_env(self):
        userstory_env = UserstoryEnv(4, 2, 2)
        backlog_env = BacklogEnv(12, 6, 6, 0, 0, 0)
        reward_system = StaircaseRewardSystem(
            config={}, coefficient=1, gamma=1, sprint_edge=self.staircase_edge_sprint
        )

        env = ProductOwnerEnv(
            userstory_env,
            backlog_env,
            with_info=True,
            reward_system=reward_system,
        )
        update_reward_system_config(env, reward_system)

        return env
