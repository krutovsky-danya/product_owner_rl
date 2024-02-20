import unittest
from environment.environment import ProductOwnerEnv
from environment.userstory_env import UserstoryEnv
from game.game_constants import GlobalConstants
import numpy as np
from environment.backlog_env import BACKLOG_COMMON_FEATURE_COUNT


IS_SILENT = False


class TestEnvFunctions(unittest.TestCase):
    def setUp(self):
        userstory_env = UserstoryEnv(userstories_common_count=4,
                                     userstories_bug_count=2,
                                     userstories_td_count=1)
        self.env = ProductOwnerEnv(userstory_env)

    def test_start_game(self):
        # тестируются действия, выполняемые с момента начала игры до первого релиза включительно
        state = self.env.reset()
        if not IS_SILENT:
            print(state)
        game_sim = self.env.game

        self.buy_statistical_research(game_sim.context.get_money(), len(game_sim.userstories.stories_list))
        self.move_userstory_card(len(game_sim.userstories.stories_list), len(game_sim.userstories.release))
        self.press_userstories_release()

        while not game_sim.hud.release_available:
            can_move = self.move_backlog_card()
            while can_move:
                can_move = self.move_backlog_card()

            self.start_sprint()

        self.release_product()
    
    def test_state_dim(self):
        state_dim = self.env.state_dim
        state = self.env.reset()

        self.assertEqual(len(state), state_dim)

    def buy_statistical_research(self, current_money, us_count):
        state = self.env.step(5)  # buy statistical research
        if not IS_SILENT:
            print(state)
        game_sim = self.env.game

        self.assertEqual(current_money - GlobalConstants.statistical_research_cost,
                         game_sim.context.get_money())
        self.assertEqual(us_count + 2, len(game_sim.userstories.stories_list))

    def move_userstory_card(self, us_story_count, us_release_count):
        state = self.env.step(7)  # move userstory card
        if not IS_SILENT:
            print(state)
        game_sim = self.env.game

        self.assertEqual(us_story_count - 1, len(game_sim.userstories.stories_list))
        self.assertEqual(us_release_count + 1, len(game_sim.userstories.release))

    def press_userstories_release(self):
        game_sim = self.env.game
        self.assertTrue(game_sim.userstories.release_available)

        state = self.env.step(1)  # start userstory release
        if not IS_SILENT:
            print(state)
        # it is actually decomposition, not release but in godot it was named release
        self.assertGreater(len(game_sim.backlog.backlog), 0)
        self.assertEqual(len(game_sim.userstories.release), 0)

    def move_backlog_card(self):
        game_sim = self.env.game
        action_index = self.find_available_to_move_backlog_card()
        if action_index is not None:
            backlog_count = len(game_sim.backlog.backlog)
            sprint_count = len(game_sim.backlog.sprint)

            state = self.env.step(action_index)  # move backlog card
            if not IS_SILENT:
                print(state)

            self.assertEqual(backlog_count - 1, len(game_sim.backlog.backlog))
            self.assertEqual(sprint_count + 1, len(game_sim.backlog.sprint))
            return self.can_move_any_backlog_card()

        return self.can_move_any_backlog_card()

    def find_available_to_move_backlog_card(self):
        state = self.env._get_state()
        backlog_begin = self.env.meta_space_dim + \
            self.env.userstory_env.userstory_space_dim
        backlog_end = backlog_begin + self.env.backlog_env.backlog_space_dim
        state = state[backlog_begin:backlog_end]
        game_sim = self.env.game

        current_hours = game_sim.backlog.calculate_hours_sum()
        hours_boundary = game_sim.context.available_developers_count * GlobalConstants.developer_hours
        for i in range(0, len(state), BACKLOG_COMMON_FEATURE_COUNT):
            card_hours = state[i]
            if card_hours + current_hours <= hours_boundary:
                return int(i / BACKLOG_COMMON_FEATURE_COUNT) + self.env.meta_action_dim + self.env.userstory_max_action_num

    def can_move_any_backlog_card(self):
        game_sim = self.env.game
        current_hours = game_sim.backlog.calculate_hours_sum()
        hours_boundary = game_sim.context.available_developers_count * GlobalConstants.developer_hours
        backlog = game_sim.backlog.backlog

        for i in range(len(backlog)):
            card = backlog[i]
            if card.info.hours + current_hours <= hours_boundary:
                return True
        return False

    def start_sprint(self):
        game_sim = self.env.game
        self.assertTrue(game_sim.backlog.can_start_sprint())

        state = self.env.step(0)  # start sprint
        if not IS_SILENT:
            print(state)
        self.assertEqual(len(game_sim.backlog.sprint), 0)

    def release_product(self):
        game_sim = self.env.game
        self.assertTrue(game_sim.hud.release_available)
        is_first_release = game_sim.is_first_release

        state = self.env.step(2)  # release product
        if not IS_SILENT:
            print(state)

        if is_first_release:
            self.assertEqual(game_sim.context.customers, 25)
            self.assertEqual(game_sim.context.get_loyalty(), 4)
            self.assertTrue(game_sim.userstories.available)
            self.assertTrue(game_sim.userstories.user_survey_available)
            self.assertTrue(game_sim.userstories.statistical_research_available)

        self.assertFalse(game_sim.is_first_release)
        self.assertFalse(game_sim.context.is_new_game)
        self.assertTrue(len(game_sim.completed_us) == 0)

    def test_act_upon_not_existing_card(self):
        state = self.env.reset()
        new_state, reward, done, _ = self.env.step(self.env.action_n - 1)

        assert np.all(state == new_state)


if __name__ == "__main__":
    unittest.main()
