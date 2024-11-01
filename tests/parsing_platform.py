import cv2
from web_interaction import GameImageParser


class ParsingPlatform:
    templates_directory = "web_interaction/templates"
    image_parser = GameImageParser(templates_directory)
    image_directory = "tests/test_images"
    original_shape = (1028, 1920, 3)

    orange = (43, 194, 249)

    def read_game_start(self, id):
        image_path = self.image_directory + f"/game_start_{id}.png"
        return cv2.imread(image_path)
