import cv2

from game import ProductOwnerGame
from game.userstory_card.userstory_card_info import UserStoryCardInfo

from .image_parser import GameImageParser
from .single_color_storage import SingleColorStorage


class GameCoordinator:
    def __init__(self, image_parser: GameImageParser) -> None:
        self.image_parser = image_parser

    def insert_user_stories_from_image(
        self, game: ProductOwnerGame, image: cv2.typing.MatLike
    ):
        user_stories = self.image_parser.read_user_stories(image)

        game.userstories.stories_list.clear()
        game.context.available_stories.clear()

        sprint = game.context.current_sprint

        for user_story in user_stories:
            color_storage = SingleColorStorage(user_story.color)
            game_user_story = UserStoryCardInfo("S", sprint, color_storage)
            game.userstories.add_us(game_user_story)
