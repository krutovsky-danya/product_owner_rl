import cv2
import numpy as np
from os import listdir, path
from typing import Tuple, List

from .user_story_image_info import UserStoryImageInfo
from .backlog_card_image_info import BacklogCardImageInfo

_DEFAULT_TEMPLATES_PATH = "web_interation/templates"
Coordinates = Tuple[int, int]
Shape = Tuple[int, int, int]
Image = cv2.typing.MatLike


class GameImageParser:
    def __init__(self, templates_path=_DEFAULT_TEMPLATES_PATH) -> None:
        self.templates_path = templates_path
        self.black = np.array([0, 0, 0])
        self.white = np.array([255, 255, 255])
        self.templates = self._load_templates()

        self.pink = (120, 79, 240)
        self.red = (54, 79, 234)
        self.orange = (23, 150, 247)
        self.yellow = (43, 194, 249)
        self.green = (115, 188, 30)
        self.blue = (255, 211, 143)
        self.purple = (243, 132, 168)

        self.board_positions = {
            (540, 960, 3): {"y_0": 135, "y_1": 495, "x_0": 715, "x_1": 925},
            (1028, 1920, 3): {"x_0": 1372, "y_0": 268, "x_1": 1750, "y_1": 939},
        }

        self.board_queue_params = {
            (1028, 1920, 3): {
                "x_left": 5,
                "x_right": 180,
                "y_upper": 65,
                "y_lower": 610,
            }
        }

        self.loyalty_nums_positions = {
            (540, 960, 3): {"x_0": 49, "y_0": 7, "y_1": 15},
            (1028, 1920, 3): {"x_0": 94, "y_0": 12, "y_1": 29},
        }

        self.customers_nums_positions = {
            (540, 960, 3): {"x_0": 49, "y_0": 19, "y_1": 27},
            (1028, 1920, 3): {"x_0": 94, "y_0": 34, "y_1": 51},
        }

        self.user_story_num_width = {
            (540, 960, 3): 6,
            (1028, 1920, 3): 11,
        }

        self.header_positions = {
            (540, 960, 3): {"x_0": 57, "y_0": 7, "x_1": 932, "y_1": 83},
            (1028, 1920, 3): {"x_0": 184, "y_0": 36, "x_1": 1794, "y_1": 136},
        }

        self.sprint_params = {
            (540, 960, 3): {"y_0": 14, "y_1": 30, "x_0": 487, "x_1": 630, "width": 11},
            (1028, 1920, 3): {
                "x_0": 901,
                "y_0": 7,
                "x_1": 1100,
                "y_1": 32,
                "width": 21,
            },
        }

        self.money_params = {
            (540, 960, 3): {"y_0": 33, "y_1": 49, "x_0": 421, "x_1": 480, "width": 11},
            (1028, 1920, 3): {
                "x_0": 750,
                "y_0": 44,
                "x_1": 900,
                "y_1": 69,
                "width": 21,
            },
        }

        self.loyalty_params = {
            (540, 960, 3): {"y_0": 38, "y_1": 49, "x_0": 143, "x_1": 206, "width": 9},
            (1028, 1920, 3): {
                "x_0": 250,
                "y_0": 44,
                "x_1": 400,
                "y_1": 75,
                "width": 18,
            },
        }

        self.customers_params = {
            (540, 960, 3): {"y_0": 18, "y_1": 29, "x_0": 161, "x_1": 206, "width": 9},
            (1028, 1920, 3): {
                "x_0": 291,
                "y_0": 9,
                "x_1": 400,
                "y_1": 40,
                "width": 18,
            },
        }

        self.cards_params = {
            (540, 960, 3): {"l": 42, "r": 46},
            (1028, 1920, 3): {"l": 81, "r": 87},
        }

        self.hours_positions = {
            (540, 960, 3): {"x_0": 3, "x_1": 25, "y_0": 9, "y_1": 24},
            (1028, 1920, 3): {"x_0": 7, "x_1": 55, "y_0": 17, "y_1": 44},
        }

        self.backlog_num_width = {
            (540, 960, 3): 11,
            (1028, 1920, 3): 21,
        }

    def _get_image_char(self, filename: str):
        if filename.startswith("empty"):
            return ""
        return filename[0]

    def _load_templates(self):
        templates: List[Tuple[str, cv2.typing.MatLike]] = []
        for template_filename in listdir(self.templates_path):
            image_char = self._get_image_char(template_filename)
            image = cv2.imread(path.join(self.templates_path, template_filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            templates.append((image_char, image))
        return templates

    def read_digit(self, image: cv2.typing.MatLike, tolerance: float = 0.02):
        best_diff = float('inf')
        best_chararcter = ""
        for character, template in self.templates:
            if template.shape != image.shape:
                continue
            images_diff = cv2.absdiff(image, template)
            unmatched_count = (images_diff != 0).sum()

            if unmatched_count == 0:
                return character

            if unmatched_count < best_diff:
                best_diff = unmatched_count
                best_chararcter = character

        if best_diff <= tolerance * image.size:
            return best_chararcter

        widht, height = image.shape
        cv2.imwrite(self.templates_path + f"/error_{widht}_{height}.png", image)
        raise Exception("Image not found in templates")

    def is_empty_vertical(self, image: cv2.typing.MatLike, x: int):
        vertical_line = image[:, x]
        unique = np.unique(vertical_line)
        return len(unique) == 1

    def is_empty_horizontal(self, image: cv2.typing.MatLike, y: int):
        horizontal_line = image[y, :]
        unique = np.unique(horizontal_line)
        return len(unique) == 1

    def find_start(self, image: cv2.typing.MatLike, x_start: int):
        x_limit = image.shape[1] - 1
        is_empty = self.is_empty_vertical(image, x_start)

        while is_empty and x_start < x_limit:
            x_start += 1
            is_empty = self.is_empty_vertical(image, x_start)

        return x_start

    def crop_image(self, image: cv2.typing.MatLike):
        y_start = 0
        y_limit = image.shape[0] - 1

        is_empty = self.is_empty_horizontal(image, y_start)
        while is_empty and y_start < y_limit:
            y_start += 1
            is_empty = self.is_empty_horizontal(image, y_start)

        y_end = image.shape[0] - 1
        is_empty = self.is_empty_horizontal(image, y_end)
        while is_empty and y_start < y_end:
            y_end -= 1
            is_empty = self.is_empty_horizontal(image, y_end)

        return image[y_start : y_end + 1]

    def split_image(self, image: cv2.typing.MatLike, char_widht: int):
        image = self.crop_image(image)
        x_start = self.find_start(image, 0)
        digits = []
        while True:
            x_end = x_start + char_widht
            if x_end >= image.shape[1]:
                break
            digit = image[:, x_start:x_end]
            digits.append(digit)
            x_start = x_end
        return digits

    def read_line(self, image: cv2.typing.MatLike, char_width: int) -> str:
        result = ""
        image = cv2.inRange(image, self.black, self.black)
        digits = self.split_image(image, char_width)
        for digit in digits:
            character = self.read_digit(digit)
            result += character

        return result

    def get_float(self, line: str):
        if line.endswith("k"):
            return float(line[:-1]) * 1000

    def get_shifted_board(self, game_image: cv2.typing.MatLike):
        if game_image.shape != (1028, 1920, 3):
            return None
        x_0 = 1463
        y_0 = 268
        x_1 = 1841
        y_1 = 939
        board = game_image[y_0:y_1, x_0:x_1]
        return board, (x_0, y_0)

    def get_board(self, image: cv2.typing.MatLike):
        position = self.board_positions[image.shape]
        x_0 = position["x_0"]
        x_1 = position["x_1"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        board = image[y_0:y_1, x_0:x_1]
        if np.all(board[:, 0] == self.white):
            return board, (x_0, y_0)
        return self.get_shifted_board(image)

    def get_board_queue(self, board: Image, original_shape: Shape):
        queue_params = self.board_queue_params[original_shape]
        queue_x_left = queue_params["x_left"]
        queue_x_right = queue_params["x_right"]
        queue_y_upper = queue_params["y_upper"]
        queue_y_lower = queue_params["y_lower"]

        queue_image = board[queue_y_upper:queue_y_lower, queue_x_left:queue_x_right]

        is_empty_verticaly = self.is_empty_vertical(queue_image, 0)
        while is_empty_verticaly:
            queue_x_left += 1
            queue_image = queue_image[:, 1:]
            if queue_image.shape[1] == 0:
                return None, -1, -1
            is_empty_verticaly = self.is_empty_vertical(queue_image, 0)

        is_empty_verticaly = self.is_empty_vertical(queue_image, -1)
        while is_empty_verticaly:
            queue_image = queue_image[:, :-1]
            is_empty_verticaly = self.is_empty_vertical(queue_image, -1)

        is_empty_horizontal = self.is_empty_horizontal(queue_image, 0)
        while is_empty_horizontal and np.all(queue_image[0, 0] == self.white):
            queue_y_upper += 1
            queue_image = queue_image[1:, :]
            is_empty_horizontal = self.is_empty_horizontal(queue_image, 0)

        return queue_image, queue_x_left, queue_y_upper

    def get_row(self, queue: Image, start_y: int):
        white = tuple(self.white)
        row_y = start_y
        while row_y < queue.shape[0] and tuple(queue[row_y, 0]) == white:
            row_y += 1

        if row_y == queue.shape[0]:
            return None, -1, -1

        row_height = 1
        while (
            row_y + row_height < queue.shape[0]
            and tuple(queue[row_y + row_height, 0]) != white
        ):
            row_height += 1

        while tuple(queue[row_y, 0]) != tuple(queue[row_y + row_height - 1, 0]):
            row_y += 1
            row_height -= 1

        row = queue[row_y : row_y + row_height]

        return row, row_y, row_height

    def get_rows(
        self, board: Image, board_position: Coordinates, original_shape: Shape
    ):
        board_x, board_y = board_position
        queue, queue_x, queue_y = self.get_board_queue(board, original_shape)

        rows = []

        if queue is None:
            return rows

        row_y = 0
        row_center_x = board_x + queue_x + queue.shape[1] // 2

        while True:
            row, row_y, row_height = self.get_row(queue, row_y)
            if row is None:
                break
            row_center_y = board_y + queue_y + row_y + row_height // 2

            rows.append((row, (row_center_x, row_center_y)))

            row_y += row_height + 1

        return rows

    def get_user_story_loaylty_image(
        self, user_story_image: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
    ):
        position = self.loyalty_nums_positions[original_shape]
        x_0 = position["x_0"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        loyalty_image = user_story_image[y_0:y_1, x_0:]
        return loyalty_image

    def get_user_story_users_image(
        self, user_story_image: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
    ):
        position = self.customers_nums_positions[original_shape]
        x_0 = position["x_0"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        customers_nums = user_story_image[y_0:y_1, x_0:]
        return customers_nums

    def read_user_story(
        self, user_story: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
    ):
        color = tuple(user_story[0, 0])
        loyalty = self.get_user_story_loaylty_image(user_story, original_shape)
        customers = self.get_user_story_users_image(user_story, original_shape)

        char_width = self.user_story_num_width[original_shape]
        loyalty = self.read_line(loyalty, char_width)
        customers = self.read_line(customers, char_width)

        loyalty = float(loyalty)
        customers = self.get_float(customers) / 1000

        return color, loyalty, customers

    def read_user_stories(self, game_image: cv2.typing.MatLike):
        board, board_position = self.get_board(game_image)
        rows = self.get_rows(board, board_position, game_image.shape)

        user_stories: List[UserStoryImageInfo] = []
        for row, position in rows:
            color, loyalty, customers = self.read_user_story(row, game_image.shape)
            user_story = UserStoryImageInfo(color, loyalty, customers, position)
            user_stories.append(user_story)

        return user_stories

    def get_header_image(self, game_image: cv2.typing.MatLike):
        position = self.header_positions[game_image.shape]
        x_0 = position["x_0"]
        x_1 = position["x_1"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        return game_image[y_0:y_1, x_0:x_1]

    def read_sprint(
        self, header: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
    ):
        position = self.sprint_params[original_shape]
        x_0 = position["x_0"]
        x_1 = position["x_1"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        width = position["width"]

        sprint = header[y_0:y_1, x_0:x_1]
        sprint_n = self.read_line(sprint, width)
        return sprint_n

    def read_current_money(
        self, header: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
    ):
        position = self.money_params[original_shape]
        x_0 = position["x_0"]
        x_1 = position["x_1"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        width = position["width"]
        money = header[y_0:y_1, x_0:x_1]

        money = self.read_line(money, width)
        return money

    def convert_gray_numbers(self, image: cv2.typing.MatLike):
        image = cv2.inRange(image, self.black, self.white - 1)
        image = cv2.cvtColor(255 - image, cv2.COLOR_GRAY2BGR)
        return image

    def read_current_loyalty(self, header: cv2.typing.MatLike, original_shape: Shape):
        position = self.loyalty_params[original_shape]
        x_0 = position["x_0"]
        x_1 = position["x_1"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        num_width = position["width"]
        loyalty = self.convert_gray_numbers(header[y_0:y_1, x_0:x_1])

        loyalty = self.read_line(loyalty, num_width)
        return loyalty

    def read_current_customers(self, header: Image, original_shape: Shape):
        position = self.customers_params[original_shape]
        x_0 = position["x_0"]
        x_1 = position["x_1"]
        y_0 = position["y_0"]
        y_1 = position["y_1"]
        num_width = position["width"]
        customers_nums = self.convert_gray_numbers(header[y_0:y_1, x_0:x_1])

        customers_value = self.read_line(customers_nums, num_width)
        return customers_value

    def split_row(
        self,
        row: cv2.typing.MatLike,
        row_center: Tuple[int, int],
        original_shape: Tuple[int, int, int],
    ):
        card_params = self.cards_params[original_shape]
        l = card_params["l"]
        r = card_params["r"]
        left: Image = row[:, :l]
        right: Image = row[:, r:]
        if right.shape[1] == 0:
            return [[left, row_center]]
        x, y = row_center
        left_center = (x - left.shape[1] // 2, y)
        if (right[0, 0] == self.white).all():
            return ([left, left_center],)
        right_center = (x + right.shape[1] // 2, y)
        return [left, left_center], [right, right_center]

    def get_backlog_card_images(self, image: cv2.typing.MatLike):
        backlog_board, board_position = self.get_board(image)

        backlog_rows = self.get_rows(backlog_board, board_position, image.shape)
        cards = []
        for row, position in backlog_rows:
            row_cards = self.split_row(row, position, image.shape)
            cards.extend(row_cards)

        return cards

    def read_backlog_card_descripton(
        self,
        card_image: cv2.typing.MatLike,
        center: Tuple[int, int],
        original_shape: Tuple[int, int, int],
    ):
        color = tuple(card_image[0, 0])

        hours_position = self.hours_positions[original_shape]
        x_0 = hours_position["x_0"]
        x_1 = hours_position["x_1"]
        y_0 = hours_position["y_0"]
        y_1 = hours_position["y_1"]
        num_width = self.backlog_num_width[original_shape]

        hours = card_image[y_0:y_1, x_0:x_1]

        hours_value = self.read_line(hours, num_width)
        hours_value = int(hours_value)

        return BacklogCardImageInfo(color, hours_value, center)

    def read_backlog(self, image: cv2.typing.MatLike):
        backlog_cards: List[BacklogCardImageInfo] = []
        cards = self.get_backlog_card_images(image)

        for card, position in cards:
            card_descripton = self.read_backlog_card_descripton(
                card, position, image.shape
            )
            backlog_cards.append(card_descripton)

        return backlog_cards

    def is_loading(self, image: cv2.typing.MatLike):
        uniform_area = image[5:155, 5:155]
        return (uniform_area == self.black).all()
