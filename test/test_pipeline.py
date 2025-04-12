from algorithms.agents_factory import DqnAgentsFactory
from environment.backlog_env import BacklogEnv
from environment.environment import ProductOwnerEnv
from environment.reward_system import BaseRewardSystem
from pipeline.base_study import BaseStudy


class TestPipeline:
    def setup_method(self):
        backlog_env = BacklogEnv(sprint_tech_debt_count=0, sprint_commons_count=0, sprint_bugs_count=0)
        reward_system = BaseRewardSystem(config={})
        self.env = ProductOwnerEnv(backlog_env=backlog_env, reward_system=reward_system,
                                   seed=None, card_picker_seed=None)

        state_dim = self.env.state_dim
        action_n = self.env.action_n
        agent_factory = DqnAgentsFactory()
        self.agent = agent_factory.create_dqn(state_dim, action_n)

        self.study = BaseStudy(self.env, self.agent, 10)

    def test_run_study_should_not_raise_error(self):
        self.study.study_agent(1, seed=None, card_picker_seed=None)
