import cv2

import matplotlib.pyplot as plt
import numpy as np

from .parsing_platform import ParsingPlatform
from web_interaction.backlog_card_image_info import BacklogCardImageInfo


class TestBacklogParsing(ParsingPlatform):
    @classmethod
    def setup_class(cls):
        cls.image_directory = super().image_directory + "/backlog_images"
        cls.backlog_1 = cv2.imread(cls.image_directory + "/game_decomposed_1.png")

        cls.center_1_1 = (1516, 384)
        cls.card_info_1 = BacklogCardImageInfo(cls.orange, 12, cls.center_1_1)

        cls.center_1_2 = (1597, 384)
        cls.card_info_2 = BacklogCardImageInfo(cls.orange, 14, cls.center_1_2)

        cls.center_2_1 = (1516, 471)
        cls.card_info_3 = BacklogCardImageInfo(cls.orange, 12, cls.center_2_1)

    def test_read_backlock_1(self):
        # arrange
        position = self.center_1_1
        card_image = cv2.imread(self.image_directory + "/backlog_card_upper_left_1.png")

        # act
        description = self.image_parser.read_backlog_card_descripton(
            card_image, position, self.original_shape
        )

        # assert
        assert description == self.card_info_1

    def test_read_backlog_2(self):
        # arrange
        position = self.center_1_2
        card_image = cv2.imread(
            self.image_directory + "/backlog_card_upper_right_1.png"
        )

        # act
        description = self.image_parser.read_backlog_card_descripton(
            card_image, position, self.original_shape
        )

        # assert
        assert description == self.card_info_2

    def test_read_backlog_3(self):
        # arrange
        position = self.center_2_1
        card_image = cv2.imread(self.image_directory + "/backlog_card_lower_left_1.png")

        # act
        description = self.image_parser.read_backlog_card_descripton(
            card_image, position, self.original_shape
        )

        # assert
        assert description == self.card_info_3

    def test_get_backlog(self):
        # arrange
        backlog = self.backlog_1.copy()
        expected_backlog_cards = [
            self.card_info_1,
            self.card_info_2,
            self.card_info_3,
        ]

        # act
        actual_cards = self.image_parser.read_backlog(backlog)

        # assert
        assert len(actual_cards) == 3
        assert actual_cards == expected_backlog_cards

    def test_two_color_backlog(self):
        # arrange
        backlog_image = cv2.imread(self.image_directory + "/backlog_two_colors.png")

        # act
        backlog_cards = self.image_parser.read_backlog(backlog_image)

        # assert
        assert backlog_cards == [
            BacklogCardImageInfo(self.image_parser.red, 17, (1516, 384)),
            BacklogCardImageInfo(self.image_parser.red, 9, (1597, 384)),
            BacklogCardImageInfo(self.image_parser.red, 7, (1516, 471)),
            BacklogCardImageInfo(self.image_parser.red, 5, (1597, 471)),

            BacklogCardImageInfo(self.image_parser.pink, 15, (1516, 559)),
            BacklogCardImageInfo(self.image_parser.pink, 11, (1597, 559)),
            BacklogCardImageInfo(self.image_parser.pink, 8, (1516, 646)),
            BacklogCardImageInfo(self.image_parser.pink, 4, (1597, 646))
        ]

    def test_single_card_parsing(self):
        # arrange
        backlog_image = cv2.imread(self.image_directory + "/single_card_backlog.png")

        # act
        backlog = self.image_parser.read_backlog(backlog_image)

        # assert
        assert backlog == [
            BacklogCardImageInfo(self.image_parser.purple, 10, (1513, 384))
        ]
