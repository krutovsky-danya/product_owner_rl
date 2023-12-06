import sys

sys.path.insert(0, '..')

import unittest
from environment.environment import ProductOwnerEnv
from game import game
from game import game_global as Global


class TestEnvFunctions(unittest.TestCase):
    def setUp(self):
        self.env = ProductOwnerEnv(
            count_common_cards=4, count_bug_cards=2, count_td_cards=1,
            count_common_userstories=4, count_bug_userstories=2, count_td_userstories=1
        )

    def test_start_game(self):
        # тестируются действия, выполняемые с момента начала игры до первого релиза включительно
        state = self.env.reset()
        print(state)

        self.buy_statistical_research(Global.get_money(), len(game.userstories.stories_list))
        self.move_userstory_card(len(game.userstories.stories_list), len(game.userstories.release))
        self.press_userstories_release()

        while not game.hud.release_available:
            can_move = self.move_backlog_card()
            while can_move:
                can_move = self.move_backlog_card()

            self.start_sprint()

        self.release_product()

    def buy_statistical_research(self, current_money, us_count):
        state = self.env.step(5)  # buy statistical research
        print(state)

        self.assertEqual(current_money - Global.statistical_research_cost, Global.get_money())
        self.assertEqual(us_count + 2, len(game.userstories.stories_list))

    def move_userstory_card(self, us_story_count, us_release_count):
        state = self.env.step(7)  # move userstory card
        print(state)

        self.assertEqual(us_story_count - 1, len(game.userstories.stories_list))
        self.assertEqual(us_release_count + 1, len(game.userstories.release))

    def press_userstories_release(self):
        self.assertTrue(game.userstories.release_available)

        state = self.env.step(1)  # start userstory release
        print(state)
        # it is actually decomposition, not release but in godot it was named release
        self.assertGreater(len(game.backlog.backlog), 0)
        self.assertEqual(len(game.userstories.release), 0)

    def move_backlog_card(self):
        action_index = self.find_available_to_move_backlog_card()
        if action_index is not None:
            backlog_count = len(game.backlog.backlog)
            sprint_count = len(game.backlog.sprint)

            state = self.env.step(action_index)  # move backlog card
            print(state)

            self.assertEqual(backlog_count - 1, len(game.backlog.backlog))
            self.assertEqual(sprint_count + 1, len(game.backlog.sprint))
            return self.can_move_any_backlog_card()

        return self.can_move_any_backlog_card()

    def find_available_to_move_backlog_card(self):
        state = self.env._get_state()
        state = state[32:]

        current_hours = game.backlog.calculate_hours_sum()
        hours_boundary = Global.available_developers_count * Global.developer_hours
        for i in range(0, len(state), 3):
            card_hours = state[i]
            if card_hours + current_hours <= hours_boundary:
                return int(i / 3) + 14

    def can_move_any_backlog_card(self):
        current_hours = game.backlog.calculate_hours_sum()
        hours_boundary = Global.available_developers_count * Global.developer_hours
        backlog = game.backlog.backlog

        for i in range(len(backlog)):
            card = backlog[i]
            if card.info.hours + current_hours <= hours_boundary:
                return True
        return False

    def start_sprint(self):
        self.assertTrue(game.backlog.can_start_sprint())

        state = self.env.step(0)  # start sprint
        print(state)
        self.assertEqual(len(game.backlog.sprint), 0)

    def release_product(self):
        self.assertTrue(game.hud.release_available)
        is_first_release = game.is_first_release

        state = self.env.step(2)  # release product
        print(state)

        if is_first_release:
            self.assertEqual(Global.customers, 25)
            self.assertEqual(Global.get_loyalty(), 4)
            self.assertTrue(game.userstories.available)
            self.assertTrue(game.userstories.user_survey_available)
            self.assertTrue(game.userstories.statistical_research_available)

        self.assertFalse(game.is_first_release)
        self.assertFalse(Global.is_new_game)
        self.assertTrue(len(game.completed_us) == 0)


if __name__ == "__main__":
    unittest.main()
