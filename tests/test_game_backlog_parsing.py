import cv2

import matplotlib.pyplot as plt
import numpy as np

from .parsing_platform import ParsingPlatform
from web_interaction import BacklogCardImageInfo


class TestBacklogParsing(ParsingPlatform):
    @classmethod
    def setup_class(cls):
        cls.image_directory = super().image_directory + "/backlog_images"
        cls.backlog_1 = cv2.imread(cls.image_directory + "/game_decomposed_1.png")

        cls.card_1_1 = (1557, 384)
        cls.center_1_1 = (1597, 419)
        cls.card_info_1 = BacklogCardImageInfo(cls.orange, 12, cls.center_1_1)

        cls.card_1_2 = (1644, 384)
        cls.center_1_2 = (1685, 419)
        cls.card_info_2 = BacklogCardImageInfo(cls.orange, 14, cls.center_1_2)

        cls.card_2_1 = (1557, 471)
        cls.center_2_1 = (1597, 506)
        cls.card_info_3 = BacklogCardImageInfo(cls.orange, 12, cls.center_2_1)

    def test_split_row_1(self):
        # arrange
        backlog = self.backlog_1.copy()
        expected_upper_left = cv2.imread(
            self.image_directory + "/backlog_card_upper_left_1.png"
        )
        expected_upper_right = cv2.imread(
            self.image_directory + "/backlog_card_upper_right_1.png"
        )
        board, board_position = self.image_parser.get_board(backlog)
        rows = self.image_parser.get_rows(board, backlog.shape, board_position)
        row, row_posiition = rows[0]

        # act
        cards = self.image_parser.split_row(row, row_posiition, backlog.shape)

        # assert
        assert len(cards) == 2
        left, right = cards
        left_image, left_position = left
        right_image, right_postion = right

        assert left_position == self.card_1_1
        image_diff = cv2.absdiff(expected_upper_left, left_image)
        assert np.all(image_diff == 0)

        assert right_postion == self.card_1_2
        image_diff = cv2.absdiff(expected_upper_right, right_image)
        assert np.all(image_diff == 0)

    def test_split_row_2(self):
        # arrange
        backlog = self.backlog_1.copy()
        expected_left = cv2.imread(
            self.image_directory + "/backlog_card_lower_left_1.png"
        )
        board, board_position = self.image_parser.get_board(backlog)
        rows = self.image_parser.get_rows(board, backlog.shape, board_position)
        row, row_posiition = rows[1]

        # act
        cards = self.image_parser.split_row(row, row_posiition, backlog.shape)

        # assert
        assert len(cards) == 1
        left = cards[0]
        left_image, left_position = left
        assert left_position == (1557, 471)
        assert left_image.shape == expected_left.shape
        image_diff = cv2.absdiff(expected_left, left_image)
        assert np.all(image_diff == 0)

    def test_get_backlog_cards_1(self):
        # arrange
        backlog = self.backlog_1.copy()
        expected_upper_left = cv2.imread(
            self.image_directory + "/backlog_card_upper_left_1.png"
        )
        expected_upper_right = cv2.imread(
            self.image_directory + "/backlog_card_upper_right_1.png"
        )
        expected_lower_left = cv2.imread(
            self.image_directory + "/backlog_card_lower_left_1.png"
        )

        expected_images = [
            expected_upper_left,
            expected_upper_right,
            expected_lower_left,
        ]
        expected_positions = [self.card_1_1, self.card_1_2, self.card_2_1]

        expected = list(zip(expected_images, expected_positions))

        # act
        backlog_cards = self.image_parser.get_backlog_card_images(backlog)

        # assert
        assert len(backlog_cards) == 3

        for i in range(3):
            actual_image, actual_pos = backlog_cards[i]
            expected_image, expected_pos = expected[i]

            assert actual_pos == expected_pos
            assert np.all(cv2.absdiff(actual_image, expected_image) == 0)
    
    def test_read_backlock_1(self):
        # arrange
        position = self.card_1_1
        card_image = cv2.imread(
            self.image_directory + "/backlog_card_upper_left_1.png"
        )

        # act
        description = self.image_parser.read_backlog_card_descripton(card_image, position, self.original_shape)

        # assert
        assert description == self.card_info_1
    def test_read_backlog_2(self):
        # arrange
        position = self.card_1_2
        card_image = cv2.imread(
            self.image_directory + "/backlog_card_upper_right_1.png"
        )

        # act
        description = self.image_parser.read_backlog_card_descripton(card_image, position, self.original_shape)

        # assert
        assert description == self.card_info_2
    
    def test_read_backlog_3(self):
        # arrange
        position = self.card_2_1
        card_image = cv2.imread(
            self.image_directory + "/backlog_card_lower_left_1.png"
        )

        # act
        description = self.image_parser.read_backlog_card_descripton(card_image, position, self.original_shape)

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
