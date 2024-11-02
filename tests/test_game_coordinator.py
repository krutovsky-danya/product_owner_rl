import cv2

from game import ProductOwnerGame
from game.backlog_card.card_info import CardInfo
from game.userstories.userstories import UserStoryCardInfo, UserStoryCard

from random import Random

from web_interaction import (
    GameImageParser,
    GameCoordinator,
    SingleColorStorage,
    BacklogCardImageInfo,
)


class TestGameCoordination:
    green = (115, 188, 30)
    orange = (43, 194, 249)
    templates_directory = "web_interaction/templates"
    image_parser = GameImageParser(templates_directory)
    random = Random(0)

    image_directory = "tests/test_images"
    initial_image_path = image_directory + "/game_start_1.png"
    initial_image = None

    @classmethod
    def setup_class(cls):
        cls.initial_image = cv2.imread(cls.initial_image_path)
        cls.backlog_image = cv2.imread(
            cls.image_directory + "/backlog_images/game_decomposed_1.png"
        )

        cls.upper_left_backlog_card_center = (1557, 419)
        cls.upper_right_backlog_card_center = (1639, 419)

    def setup_method(self):
        game = self.game = ProductOwnerGame()
        context = game.context

        context.is_new_game = False
        game.is_first_release = False
        game.userstories.disable_restrictions()
        game.office.toggle_purchases(True)

        self.game_coordinator = GameCoordinator(self.image_parser)

    def setup_orange_user_story(self):
        self.orange_color_storage = SingleColorStorage(self.orange)
        self.orange_user_story_info = UserStoryCardInfo(
            "S", 4, self.orange_color_storage, self.random
        )
        self.orange_user_story_info.related_cards.clear()
        self.game.userstories.add_us(self.orange_user_story_info)
        self.orange_user_story = self.game.userstories.stories_list[0]
        self.game.move_userstory_card(self.orange_user_story)

    def get_orange_backlog_card(self, hours):
        return CardInfo(
            hours,
            self.orange,
            id(self.orange_user_story),
            self.orange_user_story_info.label,
            self.orange_user_story_info.card_type,
        )

    def test_insert_user_stories(self):
        # arrange
        initial_image = self.initial_image.copy()
        color_storage = SingleColorStorage(self.green)
        user_story = UserStoryCardInfo("S", 4, color_storage, self.random)
        user_story.loyalty = 0.045
        user_story.customers_to_bring = 1.0
        expected_user_stories = [UserStoryCard(user_story)]

        # act
        self.game_coordinator.insert_user_stories_from_image(self.game, initial_image)

        # assert
        assert self.game.userstories.stories_list == expected_user_stories

    def test_update_header_info(self):
        # arrange
        initial_image = self.initial_image.copy()

        # act
        self.game_coordinator.update_header_info(self.game, initial_image)

        # assert
        assert self.game.context.current_sprint == 4
        assert self.game.context.get_money() == 33000
        assert self.game.context.get_loyalty() == 4.0
        assert self.game.context.customers == 25.0

    def test_find_user_story(self):
        # arrange
        initial_image = self.initial_image.copy()
        color_storage = SingleColorStorage(self.green)
        card_info = UserStoryCardInfo("S", 4, color_storage, self.random)
        card_info.loyalty = 0.045
        card_info.customers_to_bring = 1.0
        user_story = UserStoryCard(card_info)

        # act
        self.game_coordinator.insert_user_stories_from_image(self.game, initial_image)
        position = self.game_coordinator.find_user_story_position(user_story)

        # assert
        assert position == (1466, 384)

    def test_insert_backlog_cards(self):
        # arrange
        backlog_image = cv2.imread(
            self.image_directory + "/backlog_images/game_decomposed_1.png"
        )
        self.setup_orange_user_story()
        user_story_info = self.orange_user_story_info

        # act
        self.game_coordinator.insert_backlog_cards_from_image(self.game, backlog_image)

        # assert
        related_cards = user_story_info.related_cards
        assert len(related_cards) == 3
        assert related_cards == [
            self.get_orange_backlog_card(12),
            self.get_orange_backlog_card(14),
            self.get_orange_backlog_card(12),
        ]

    def test_find_backlog_card_position(self):
        # arrange
        backlog_image = self.backlog_image.copy()
        self.setup_orange_user_story()

        self.game_coordinator.insert_backlog_cards_from_image(self.game, backlog_image)
        backlog_card = self.get_orange_backlog_card(14)

        # act
        actual_position = self.game_coordinator.find_backlog_card_position(backlog_card)

        # assert
        assert actual_position == self.upper_right_backlog_card_center

    def test_remove_backlog_card(self):
        # arrange
        backlog_image = self.backlog_image.copy()
        self.setup_orange_user_story()
        self.game_coordinator.insert_backlog_cards_from_image(self.game, backlog_image)
        backlog_card = self.get_orange_backlog_card(12)

        # act
        self.game_coordinator.remove_backlog_card_from_backlog(backlog_card)

        # assert
        backlog_cards = self.game_coordinator.backlog_cards
        assert len(backlog_cards) == 2
        assert backlog_cards == [
            BacklogCardImageInfo(self.orange, 14, self.upper_left_backlog_card_center),
            BacklogCardImageInfo(self.orange, 12, self.upper_right_backlog_card_center),
        ]
