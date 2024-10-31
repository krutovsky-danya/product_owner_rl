import cv2

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
    initial_image_path = image_directory + "/game_start.png"

    def setup_method(self):
        game = self.game = ProductOwnerGame()
        context = game.context

        context.is_new_game = False
        game.is_first_release = False
        game.userstories.disable_restrictions()
        game.office.toggle_purchases(True)

    def test_insert_user_stories(self):
        # arrange
        initial_image = cv2.imread(self.initial_image_path)
        color_storage = SingleColorStorage((115, 188, 30))
        user_story = UserStoryCardInfo("S", 4, color_storage, self.random)
        user_story.loyalty = 0.045
        user_story.customers_to_bring = 1.0
        expected_user_stories = [UserStoryCard(user_story)]

        # act
        self.game_coordinator.insert_user_stories_from_image(self.game, initial_image)

        # assert
        assert self.game.userstories.stories_list == expected_user_stories
