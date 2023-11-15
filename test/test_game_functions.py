import unittest
from game import game
from game import game_global as Global


class TestGameFunctions(unittest.TestCase):
    def setUp(self):
        game.load_game()

    def test_availability_on_start(self):
        self.assertTrue(game.userstories.statistical_research_available)
        self.assertFalse(game.userstories.user_survey_available)

        # не должно выдать ошибку
        game.move_backlog_card(0)
        game.move_userstory_card(0)

        # до выпуска MVP нельзя купить комнату или робота
        workers = Global.available_developers_count
        start_money = Global.get_money()
        game.buy_room(1)
        self.assertEqual(workers, Global.available_developers_count)
        self.assertEqual(start_money, Global.get_money())
        game.buy_robot(0)
        self.assertEqual(workers, Global.available_developers_count)
        self.assertEqual(start_money, Global.get_money())

        self.assertFalse(game.hud.release_available)
        self.assertFalse(game.userstories.release_available)
        self.assertFalse(game.backlog.can_start_sprint())

        # не должно выдать ошибку
        game.load_game()

    def test_start_game(self):
        # тестируются действия, выполняемые с момента начала игры до первого релиза включительно

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
        game.press_statistical_research()

        self.assertEqual(current_money - Global.statistical_research_cost, Global.get_money())
        self.assertEqual(us_count + 2, len(game.userstories.stories_list))

    def move_userstory_card(self, us_story_count, us_release_count):
        game.move_userstory_card(0)

        self.assertEqual(us_story_count - 1, len(game.userstories.stories_list))
        self.assertEqual(us_release_count + 1, len(game.userstories.release))

    def press_userstories_release(self):
        self.assertTrue(game.userstories.release_available)

        game.userstories_start_release()
        self.assertGreater(len(game.backlog.backlog), 0)
        self.assertEqual(len(game.userstories.release), 0)

    def move_backlog_card(self):
        card_index = self.find_available_to_move_backlog_card()
        if card_index is not None:
            backlog_count = len(game.backlog.backlog)
            sprint_count = len(game.backlog.sprint)

            game.move_backlog_card(card_index)

            self.assertEqual(backlog_count - 1, len(game.backlog.backlog))
            self.assertEqual(sprint_count + 1, len(game.backlog.sprint))
            return True

        return False

    def find_available_to_move_backlog_card(self):
        backlog = game.backlog.backlog
        current_hours = game.backlog.calculate_hours_sum()
        hours_boundary = Global.available_developers_count * Global.developer_hours
        for i in range(len(backlog)):
            card = backlog[i]
            if card.info.hours + current_hours <= hours_boundary:
                return i

    def start_sprint(self):
        self.assertTrue(game.backlog.can_start_sprint())

        game.backlog_start_sprint()
        self.assertEqual(len(game.backlog.sprint), 0)

    def release_product(self):
        self.assertTrue(game.hud.release_available)
        is_first_release = game.is_first_release

        game.hud_release_product()

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
