from environment.reward_sytem import BaseRewardSystem
from environment.reward_sytem import EmpiricalRewardSystem, EmpiricalCreditStageRewardSystem
from environment.reward_sytem import FullPotentialCreditRewardSystem
from pipeline.logging_study import LoggingStudy
from environment import TutorialSolverEnv, CreditPayerEnv, ProductOwnerEnv
from environment.credit_payer_env import USUAL_CREDIT_ENV_END_SPRINT, EARLY_CREDIT_ENV_END_SPRINT

from typing import Dict, Optional, List

STUDY = "study"
TUTORIAL = "tutorial"
CREDIT_FULL = "credit"
CREDIT_START = "credit start"
CREDIT_END = "credit end"
END = "end"


def update_reward_system_config(env: ProductOwnerEnv, reward_system: BaseRewardSystem):
    backlog = env.backlog_env
    offset = env.meta_action_dim + env.userstory_env.max_action_num + backlog.backlog_max_action_num
    actions = [offset + i for i in range(backlog.sprint_max_action_num)]
    reward_system.config["remove_sprint_card_actions"] = actions


class AggregatorStudy(LoggingStudy):
    def __init__(self, environments: Dict, agents: Dict, order: List[str],
                 trajectory_max_len, save_rate=None, backlog_environments: Optional[Dict] = None,
                 userstory_environments: Optional[Dict] = None,
                 reward_systems: Optional[Dict] = None) -> None:
        # предполагается, что STUDY идет после всех элементов из order
        assert STUDY in environments
        assert STUDY in agents
        for name in order:
            assert name in agents and (reward_systems is None or name in reward_systems)
        self.environments = environments
        self.agents = agents
        self.order = order
        self.backlog_environments = backlog_environments
        if backlog_environments is None:
            self.backlog_environments = {}
        self.userstory_environments = userstory_environments
        if userstory_environments is None:
            self.userstory_environments = {}
        self.reward_systems = reward_systems
        super().__init__(environments[STUDY], agents[STUDY], trajectory_max_len, save_rate)

    def play_trajectory(self, state, info, init_discount=1):
        full_reward = 0
        full_discounted_reward = 0
        discount = init_discount

        for name in self.order:
            translator_env = self.get_translator_env(name)
            init_done = self.get_initial_done(name)
            update_reward_system_config(translator_env, translator_env.reward_system)
            state, info, reward, failed, discounted_reward, discount = self.play_some_stage(
                self.agents[name],
                translator_env,
                init_done,
                name,
                discount
            )
            full_reward += reward
            full_discounted_reward += discounted_reward
            if failed:
                self._log_trajectory_end(full_reward)
                return full_reward, full_discounted_reward
        reward_study, discounted_reward_study = super().play_trajectory(state, info, discount)
        full_reward += reward_study
        full_discounted_reward += discounted_reward_study
        self.logger.info(f"full total_reward: {full_reward}")
        return full_reward, full_discounted_reward

    def play_some_stage(self, agent, translator_env, init_done, name, init_discount):
        agent.eval()
        translator_env.game = self.env.game
        done = init_done
        state = translator_env.recalculate_state()
        info = translator_env.get_info()
        inner_sprint_action_count = 0
        total_reward = 0
        total_discounted_reward = 0
        discount = init_discount

        while not done:
            action = agent.get_action(state, info)
            action, inner_sprint_action_count = self._choose_action(action,
                                                                    inner_sprint_action_count)
            state, reward, done, info = translator_env.step(action)

            self._log_after_action(action)

            total_reward += reward
            total_discounted_reward += reward * discount
            discount *= agent.gamma

        self.logger.debug(f"{name} end")
        if translator_env.game.context.get_money() < 0:
            self.logger.debug(f"{name} failed")

        state = self.env.recalculate_state()
        info = self.env.get_info()
        failed = translator_env.game.context.get_money() < 0

        return state, info, total_reward, failed, total_discounted_reward, discount

    def get_translator_env(self, name):
        if name in self.environments:
            return self.environments[name]

        backlog_env = self.get_backlog_env(name)
        userstory_env = self.get_userstory_env(name)
        reward_system = self.get_reward_system(name)

        if name == TUTORIAL:
            return TutorialSolverEnv(backlog_env=backlog_env,
                                     userstory_env=userstory_env,
                                     with_info=self.env.with_info,
                                     reward_system=reward_system,
                                     seed=None,
                                     card_picker_seed=None)
        if name == CREDIT_FULL or name == CREDIT_END:
            return CreditPayerEnv(userstory_env=userstory_env,
                                  backlog_env=backlog_env,
                                  with_end=True,
                                  with_info=self.env.with_info,
                                  reward_system=reward_system,
                                  seed=None,
                                  card_picker_seed=None)
        if name == CREDIT_START:
            return CreditPayerEnv(userstory_env=userstory_env,
                                  backlog_env=backlog_env,
                                  with_end=False,
                                  with_info=self.env.with_info,
                                  reward_system=reward_system,
                                  seed=None,
                                  card_picker_seed=None)
        return self.env

    def get_backlog_env(self, name):
        if name in self.backlog_environments:
            return self.backlog_environments[name]

        return None

    def get_userstory_env(self, name):
        if name in self.userstory_environments:
            return self.userstory_environments[name]

        return None

    def get_reward_system(self, name):
        if not (self.reward_systems is None):
            return self.reward_systems[name]

        if name == TUTORIAL:
            return EmpiricalRewardSystem(config={})
        if name == CREDIT_FULL or name == CREDIT_START:
            return FullPotentialCreditRewardSystem(config={})
        if name == CREDIT_END:
            return EmpiricalCreditStageRewardSystem(with_late_purchase_punishment=True, config={})

        return EmpiricalRewardSystem(config={})

    def get_initial_done(self, name):
        info = self.env.get_info()
        game_ended = self.env.get_done(info)
        if game_ended:
            return game_ended
        if name == TUTORIAL:
            return not self.env.game.context.is_new_game
        if name == CREDIT_FULL or name == CREDIT_END:
            return self.env.game.context.current_sprint >= USUAL_CREDIT_ENV_END_SPRINT
        if name == CREDIT_START:
            return self.env.game.context.current_sprint >= EARLY_CREDIT_ENV_END_SPRINT
