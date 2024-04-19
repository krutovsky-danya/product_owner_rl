from environment.userstory_env import UserstoryEnv
from environment.backlog_env import BacklogEnv
from environment.reward_sytem import BaseRewardSystem, EmpiricalRewardSystem, EmpiricalCreditStageRewardSystem, FullPotentialCreditRewardSystem
from pipeline.logging_study import LoggingStudy
from environment import TutorialSolverEnv, CreditPayerEnv, ProductOwnerEnv
from environment.credit_payer_env import USUAL_CREDIT_ENV_END_SPRINT, EARLY_CREDIT_ENV_END_SPRINT


def update_reward_system_config(env: ProductOwnerEnv, reward_system: BaseRewardSystem):
    backlog = env.backlog_env
    offset = env.meta_action_dim + env.userstory_env.max_action_num + backlog.backlog_max_action_num
    actions = [offset + i for i in range(backlog.sprint_max_action_num)]
    reward_system.config["remove_sprint_card_actions"] = actions

class AggregatorStudy(LoggingStudy):
    def __init__(self, env, agents, trajectory_max_len, save_rate=None,
                 backlog_environments=None, reward_systems=None) -> None:
        assert 0 < len(agents) < 5
        assert backlog_environments is None or len(backlog_environments) == len(agents) - 1
        self.stage = len(agents)
        if self.stage == 1:
            assert isinstance(env, TutorialSolverEnv)
        if self.stage == 2:
            assert isinstance(env, CreditPayerEnv)
        if self.stage == 3:
            assert isinstance(env, CreditPayerEnv) and env.with_end
        if self.stage == 4:
            assert isinstance(env, ProductOwnerEnv)

        self.agents = agents
        self.backlog_environments = backlog_environments
        if backlog_environments is None:
            self.backlog_environments = [None] * self.stage
        self.rewards_systems = reward_systems
        super().__init__(env, agents[-1], trajectory_max_len, save_rate)

    def play_trajectory(self, state, info):
        full_reward = 0
        if self.stage > 1:
            if self.rewards_systems is None or len(self.rewards_systems) == 0:
                tutorial_reward_system = EmpiricalRewardSystem(config={})
            else:
                tutorial_reward_system = self.rewards_systems[0]
            state, info, reward, failed = self.play_tutorial(self.agents[0],
                                                             self.backlog_environments[0],
                                                             tutorial_reward_system)
            full_reward += reward
            if failed:
                self._log_trajectory_end(full_reward)
                return reward
        if self.stage > 2:
            if self.rewards_systems is None or len(self.rewards_systems) < 2:
                credit_reward_system = FullPotentialCreditRewardSystem(config={})
            else:
                credit_reward_system = self.rewards_systems[1]
            state, info, credit_reward, failed = self.play_credit_payment(self.agents[1],
                                                                          self.backlog_environments[1],
                                                                          False,
                                                                          credit_reward_system)
            full_reward += credit_reward
            if failed:
                self._log_trajectory_end(full_reward)
                return full_reward
        if self.stage > 3:
            if self.rewards_systems is None or len(self.rewards_systems) < 3:
                credit_reward_system = FullPotentialCreditRewardSystem(config={})
            else:
                credit_reward_system = self.rewards_systems[2]
            state, info,  credit_reward, failed = self.play_credit_payment(self.agents[2],
                                                                           self.backlog_environments[2],
                                                                           True,
                                                                           credit_reward_system)
            full_reward += credit_reward
            if failed:
                self._log_trajectory_end(full_reward)
                return full_reward
        full_reward += super().play_trajectory(state, info)
        self.logger.info(f"full total_reward: {full_reward}")
        return full_reward

    def play_tutorial(self, tutorial_agent, tutorial_backlog_env, tutorial_reward_system):
        env = TutorialSolverEnv(backlog_env=tutorial_backlog_env, with_info=self.env.with_info, reward_system=tutorial_reward_system)
        update_reward_system_config(env, tutorial_reward_system)
        done = not self.env.game.context.is_new_game

        return self.play_some_stage(tutorial_agent, env, done, "tutorial")

    def play_credit_payment(self, credit_agent, credit_backlog_env, with_end, reward_system):
        env = CreditPayerEnv(backlog_env=credit_backlog_env, with_end=with_end,
                             with_info=self.env.with_info, reward_system=reward_system)
        end_sprint = USUAL_CREDIT_ENV_END_SPRINT if with_end else EARLY_CREDIT_ENV_END_SPRINT
        done = self.env.game.context.current_sprint == end_sprint
        update_reward_system_config(env, reward_system)

        return self.play_some_stage(credit_agent, env, done, "credit")

    def play_some_stage(self, agent, translator_env, init_done, name):
        agent.epsilon = 0
        translator_env.game = self.env.game
        done = init_done
        state = translator_env.recalculate_state()
        info = translator_env.get_info()
        inner_sprint_action_count = 0
        total_reward = 0

        while not done:
            action = agent.get_action(state, info)
            action, inner_sprint_action_count = self._choose_action(action,
                                                                    inner_sprint_action_count)
            state, reward, done, info = translator_env.step(action)

            self._log_after_action(action)

            total_reward += reward

        self.logger.debug(f"{name} end")
        if translator_env.game.context.get_money() < 0:
            self.logger.debug(f"{name} failed")

        return self.env.recalculate_state(), self.env.get_info(), \
               total_reward, translator_env.game.context.get_money() < 0
