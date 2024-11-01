import cv2

from random import Random

from game import ProductOwnerGame
from game.userstory_card.userstory_card import UserStoryCard
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

    def update_header_info(
        self, game: ProductOwnerGame, game_image: cv2.typing.MatLike
    ):
        shape = game_image.shape
        header = self.image_parser.get_header_image(game_image)

        customers = self.image_parser.read_current_customers(header, shape)
        loyalty = self.image_parser.read_current_loyalty(header, shape)

        sprint = self.image_parser.read_sprint(header, shape)
        money = self.image_parser.read_current_money(header, shape)

        game.context.customers = float(customers) / 1000
        game.context.set_loyalty(float(loyalty))

        game.context.current_sprint = int(sprint)
        game.context.set_money(float(money.removesuffix("$")))

    def find_user_story_position(self, user_story: UserStoryCard):
        for element in self.user_stories:
            if element == user_story:
                return element.position
        raise Exception(f"Not found user stroy {user_story}")
