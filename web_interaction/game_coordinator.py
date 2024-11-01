import cv2

from random import Random

from game import ProductOwnerGame
from game.userstory_card.userstory_card_info import UserStoryCardInfo

from .image_parser import GameImageParser
from .single_color_storage import SingleColorStorage


class GameCoordinator:
    def __init__(self, image_parser: GameImageParser) -> None:
        self.image_parser = image_parser
        self.random = Random(0)
        self.user_stories = []

    def skip_tutorial(self, game: ProductOwnerGame):
        context = game.context

        context.is_new_game = False
        game.is_first_release = False
        game.userstories.disable_restrictions()
        game.office.toggle_purchases(True)

        game.context.color_storage = SingleColorStorage(None)

    def insert_user_stories_from_image(
        self, game: ProductOwnerGame, image: cv2.typing.MatLike
    ):
        self.user_stories = self.image_parser.read_user_stories(image)

        game.userstories.stories_list.clear()
        game.context.available_stories.clear()

        sprint = game.context.current_sprint

        for user_story in self.user_stories:
            color_storage = SingleColorStorage(user_story.color)
            game_user_story = UserStoryCardInfo("S", sprint, color_storage, self.random)
            game_user_story.loyalty = user_story.loyalty
            game_user_story.customers_to_bring = user_story.customers
            game.userstories.add_us(game_user_story)
