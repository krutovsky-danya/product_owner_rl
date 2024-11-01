import cv2

import web_interaction

import numpy as np

from game import ProductOwnerGame

from web_interaction import GameImageParser, UserStoryImageInfo


class TestInitialGameParsing:
    templates_directory = "web_interaction/templates"
    image_parser = GameImageParser(templates_directory)

    image_directory = "tests/test_images"
    initial_image_path = image_directory + "/game_start_1.png"
    _initial_image = cv2.imread(initial_image_path)

    expected_board_path = image_directory + "/expected_user_story_board.png"
    _expected_board = cv2.imread(expected_board_path)

    expected_user_story_path = image_directory + "/initial_user_story.png"
    _expected_user_story = cv2.imread(expected_user_story_path)
    expected_user_story_color = (115, 188, 30)
    expected_user_story_position = (1557, 384)

    expected_user_story_loyalty_path = (
        image_directory + "/expected_user_story_loyalty.png"
    )
    _expected_user_story_loyalty = cv2.imread(expected_user_story_loyalty_path)

    expecrted_user_story_customers_path = (
        image_directory + "/expected_user_story_users.png"
    )
    _expected_user_story_customers = cv2.imread(expecrted_user_story_customers_path)

    def setup_method(self):
        self.original_shape = (1028, 1920, 3)

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
        expected_row = self._expected_user_story.copy()

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
        user_stoty_image = self._expected_user_story.copy()
        expected_loyalty_line = self._expected_user_story_loyalty.copy()

        # act
        actual_loyalty = self.image_parser.get_user_story_loaylty_image(
            user_stoty_image, self.original_shape
        )

        # assert
        assert actual_loyalty.shape == expected_loyalty_line.shape
        image_diff = cv2.absdiff(actual_loyalty, expected_loyalty_line)
        assert np.all(image_diff == 0)

    def test_read_user_story_loyalty(self):
        # arrange
        loyalty_line_image = self._expected_user_story_loyalty.copy()

        # act
        line = self.image_parser.read_line(loyalty_line_image, 11)

        # assert
        assert line == "+0.045"

    def test_select_user_story_users(self):
        # arrange
        user_stroy_image = self._expected_user_story.copy()
        expected_users_line = self._expected_user_story_customers.copy()

        # act
        actual_users = self.image_parser.get_user_story_users_image(
            user_stroy_image, self.original_shape
        )

        assert actual_users.shape == expected_users_line.shape
        image_diff = cv2.absdiff(actual_users, expected_users_line)
        assert np.all(image_diff == 0)

    def test_read_user_story_customers(self):
        # arrange
        customers_line = self._expected_user_story_customers.copy()

        # act
        line = self.image_parser.read_line(customers_line, 11)

        # assert
        assert line == "+1000"

    def test_read_user_story(self):
        # arrange
        user_story_image = self._expected_user_story.copy()

        # act
        user_story_info = self.image_parser.read_user_story(
            user_story_image, self.original_shape
        )

        # assert
        assert user_story_info == (self.expected_user_story_color, 0.045, 1.0)

    def test_read_initial_user_stories(self):
        # arrange
        game_state = self._initial_image.copy()
        expected_user_story = UserStoryImageInfo(
            self.expected_user_story_color,
            0.045,
            1.0,
            self.expected_user_story_position,
        )

        # act
        user_stories = self.image_parser.read_user_stories(game_state)

        # assert
        assert user_stories == [expected_user_story]
