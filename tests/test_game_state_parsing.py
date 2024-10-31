import cv2

import web_interaction

import numpy as np

from game import ProductOwnerGame

from web_interaction import GameImageParser


class TestInitialGameParsing:
    templates_directory = "web_interaction/templates"
    image_parser = GameImageParser(templates_directory)

    image_directory = "tests/test_images"
    initial_image_path = image_directory + "/game_start.png"
    _initial_image = cv2.imread(initial_image_path)

    game = ProductOwnerGame()

    expected_board_path = image_directory + "/expected_user_story_board.png"
    _expected_board = cv2.imread(expected_board_path)

    expected_row_path = image_directory + "/initial_user_story.png"
    _expected_row = cv2.imread(expected_row_path)

    expected_user_story_loyalty_path = image_directory + '/expected_user_story_loyalty.png'
    _expected_user_story_loyalty = cv2.imread(expected_user_story_loyalty_path)

    def setup_method(self):
        self.original_shape = (1028, 1920, 3)

        self.game = ProductOwnerGame()

    def test_image_parser_loads_images(self):
        assert len(self.image_parser.templates) > 0

    def test_image_parser_selects_user_story_board(self):
        # arrange
        initial_image = self._initial_image.copy()
        expected_board = self._expected_board.copy()

        # act
        actual_board = self.image_parser.get_board(initial_image)

        # assert
        assert actual_board.size == expected_board.size
        images_diff = cv2.absdiff(actual_board, expected_board)
        assert np.all(images_diff == 0)

    def test_image_parser_selects_row(self):
        # arrange
        original_shape = (1028, 1920, 3)
        board = self._expected_board.copy()
        expected_row = self._expected_row.copy()

        # act
        rows = self.image_parser.get_rows(board, original_shape)

        # assert
        assert len(rows) == 1
        row, position = rows[0]
        assert row.shape == (70, 170, 3)
        assert position == (1557, 384)
        images_diff = cv2.absdiff(row, expected_row)
        assert np.all(images_diff == 0)
    
    def test_image_parser_selects_user_strory_loyalty(self):
        # arrange
        user_stoty_image = self._expected_row.copy()
        expected_loyalty_line = self._expected_user_story_loyalty.copy()

        # act
        actual_loyalty = self.image_parser.get_user_story_loaylty_image(user_stoty_image, self.original_shape)

        # assert
        assert actual_loyalty.shape == expected_loyalty_line.shape
        image_diff = cv2.absdiff(actual_loyalty, expected_loyalty_line)
        assert np.all(image_diff == 0)

    def test_read_user_story_loyalty(self):
        # arrange
        loyalty_line_image = self._expected_user_story_loyalty.copy()

        # act
        cv2.imwrite(self.image_directory + '/loyalty_line_image.png', loyalty_line_image)
        line = self.image_parser.read_line(loyalty_line_image, 11, 6, 0)

        # assert
        assert line == '+0.045'

    # def test_user_stories_parsing(self):
    #     web_interaction.insert_user_stories_from_image(self.game, self.initial_image)
