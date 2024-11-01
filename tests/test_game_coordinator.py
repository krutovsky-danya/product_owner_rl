import cv2
import pytest

from game import ProductOwnerGame
from game.userstories.userstories import UserStoryCardInfo, UserStoryCard

from random import Random

from web_interaction import GameImageParser, GameCoordinator, SingleColorStorage


class TestGameCoordination:
    templates_directory = "web_interaction/templates"
    image_parser = GameImageParser(templates_directory)
    game_coordinator = GameCoordinator(image_parser)
    random = Random(0)

    image_directory = "tests/test_images"
    initial_image_path = image_directory + "/game_start_1.png"
    initial_image = None

    @classmethod
    def setup_class(cls):
        cls.initial_image = cv2.imread(cls.initial_image_path)

    def setup_method(self):
        game = self.game = ProductOwnerGame()
        context = game.context

        context.is_new_game = False
        game.is_first_release = False
        game.userstories.disable_restrictions()
        game.office.toggle_purchases(True)

    def test_insert_user_stories(self):
        # arrange
        initial_image = self.initial_image.copy()
        color_storage = SingleColorStorage((115, 188, 30))
        user_story = UserStoryCardInfo("S", 4, color_storage, self.random)
        user_story.loyalty = 0.045
        user_story.customers_to_bring = 1.0
        expected_user_stories = [UserStoryCard(user_story)]

        # act
        self.game_coordinator.insert_user_stories_from_image(self.game, initial_image)

        # assert
        assert self.game.userstories.stories_list == expected_user_stories

    def test_update_(self):
        # arrange
        initial_image = self.initial_image.copy()

        # act
        self.game_coordinator.update_header_info(self.game, initial_image)

        # assert
        assert self.game.context.current_sprint == 4
        assert self.game.context.get_money() == 33000
        assert self.game.context.get_loyalty() == 4.0
        assert self.game.context.customers == 25.0
