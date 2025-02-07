import cv2

from itertools import groupby
from logging import Logger
from operator import attrgetter
from random import Random
from typing import Dict, List, Tuple

from game import ProductOwnerGame
from game.backlog_card.card_info import CardInfo
from game.userstory_card.userstory_card import UserStoryCard
from game.userstory_card.userstory_card_info import UserStoryCardInfo

from .image_parser import GameImageParser
from .single_color_storage import SingleColorStorage
from .backlog_card_image_info import BacklogCardImageInfo


class GameCoordinator:
    def __init__(self, image_parser: GameImageParser) -> None:
        self.image_parser = image_parser
        self.random = Random(0)
        self.user_stories = []
        self.backlog_cards = []

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

        game.context.customers = self.image_parser.get_float(customers) / 1000
        game.context.set_loyalty(float(loyalty))

        game.context.current_sprint = int(sprint)
        money = money.removesuffix("$")
        money = self.image_parser.get_float(money)
        game.context.set_money(money)

    def find_user_story_position(self, user_story: UserStoryCard):
        for element in self.user_stories:
            if element == user_story:
                return element.position
        raise Exception(f"Not found user stroy {user_story}")

    def insert_backlog_cards_from_image(
        self, game: ProductOwnerGame, game_image: cv2.typing.MatLike
    ):
        self.backlog_cards = self.image_parser.read_backlog(game_image)

        backlog_cards_by_color: Dict[Tuple, List[BacklogCardImageInfo]] = {}
        for key, group in groupby(self.backlog_cards, attrgetter("color")):
            backlog_cards_by_color[key] = list(group)

        for user_story in game.userstories.release:
            info = user_story.info
            us_id_val = id(info)
            info.related_cards.clear()
            color = info.color
            backlog_cards = backlog_cards_by_color[color]

            for backlog_card in backlog_cards:
                card_info = CardInfo(
                    hours_val=backlog_card.hours,
                    color_val=color,
                    us_id_val=us_id_val,
                    label_val=info.label,
                    card_type_val=info.card_type,
                )
                info.related_cards.append(card_info)

    def find_backlog_card_position(self, backlog_card: CardInfo):
        for saved_card in self.backlog_cards:
            if saved_card == backlog_card:
                return saved_card.position
        raise Exception(f"Not found backlog card: {backlog_card}")

    def remove_backlog_card_from_backlog(self, backlog_card: CardInfo):
        index = 0
        position = None
        for i, image_card in enumerate(self.backlog_cards):
            if image_card != backlog_card:
                continue
            index = i
            position = image_card.position
            break

        for image_card in self.backlog_cards[index + 1 :]:
            image_card.position, position = position, image_card.position

        self.backlog_cards.pop(index)

    def clear_backlog_sprint(self, game: ProductOwnerGame):
        backlog = game.backlog
        backlog.backlog.extend(backlog.sprint)
        backlog.sprint.clear()

    def log_game_state(self, game: ProductOwnerGame, logger: Logger):
        logger.info(f"Sprint {game.context.current_sprint}")
        logger.info(f"Money {game.context.get_money()}")
        logger.info(f"Loyalty {game.context.get_loyalty()}")
        logger.info(f"Customers {game.context.customers}")
        logger.info(f"Credit {game.context.credit}")

        logger.info("Current sprint hours: %d", game.context.current_sprint_hours)

        logger.info(f"Available user stories: {game.userstories.stories_list}")
        logger.info(f"User stroies to release: {game.userstories.release}")

        logger.info(f"Backlog cards: {game.backlog.backlog}")
        logger.info(f"Sprint cards: {game.backlog.sprint}")

        logger.info("Sprints without releases: %d", game.context.blank_sprint_counter)
