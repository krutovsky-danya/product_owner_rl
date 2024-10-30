import cv2

import numpy as np

from game import ProductOwnerGame

from web_interaction import GameImageParser


class TestInitialGameParsing:
    templates_directory = 'web_interaction/templates'
    image_parser = GameImageParser(templates_directory)

    image_directory = "tests/test_images"
    initial_image_path = image_directory + "/game_start.png"
    _initial_image = cv2.imread(initial_image_path)

    game = ProductOwnerGame()

    expected_board_path = image_directory + '/expected_user_story_board.png'

    def setup_method(self):
        self.initial_image = self._initial_image.copy()

        self.game = ProductOwnerGame()
    
    def test_image_parser_loads_images(self):
        assert len(self.image_parser.templates) > 0
    
    def test_image_parser_selects_user_story_board(self):
        actual_board = self.image_parser.get_board(self.initial_image)
        expected_board = cv2.imread(self.expected_board_path)
        assert actual_board.size == expected_board.size
        images_diff = cv2.absdiff(actual_board, expected_board)
        assert np.all(images_diff == 0)


    # def test_user_stories_parsing(self):
    #     web_interaction.insert_user_stories_from_image(self.game, self.initial_image)
