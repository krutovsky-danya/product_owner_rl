import cv2

import numpy as np

from .parsing_platform import ParsingPlatform


class TestHeaderParsing(ParsingPlatform):
    def setup_method(self):
        self.expected_header = cv2.imread(
            self.image_directory + "/expected_game_header.png"
        )

    def test_read_header(self):
        # arrange
        game_image = self.read_game_start(1)
        expected_header = self.expected_header.copy()

        # act
        actual_header = self.image_parser.get_header_image(game_image)

        # assert
        image_diff = cv2.absdiff(expected_header, actual_header)
        assert np.all(image_diff == 0)
    
    def test_read_sprint(self):
        # arrange
        header_image = self.expected_header.copy()

        # act
        actual_sprint = self.image_parser.get_sprint_number(header_image, self.original_shape)

        # assert
        assert actual_sprint == '4'
