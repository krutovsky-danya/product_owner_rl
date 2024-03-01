from environment.backlog_env import BacklogEnv
from pipeline.base_study import BaseStudy
from algorithms.deep_q_networks import DQN
from environment.environment import ProductOwnerEnv
import unittest


class TestPipeline(unittest.TestCase):
    def setUp(self):
        backlog_env = BacklogEnv(sprint_tech_debt_count=0, sprint_commons_count=0, sprint_bugs_count=0)
        self.env = ProductOwnerEnv(backlog_env=backlog_env)

        state_dim = self.env.state_dim
        action_n = self.env.action_n
        self.agent = DQN(state_dim, action_n)

        self.study = BaseStudy(self.env, self.agent, 1_000)

    def test_run_study_should_not_raise_error(self):
        self.study.study_agent(1)
