import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, getcwd
from typing import Tuple, List

from game.userstory_card.userstory_card import UserStoryCard
from .backlog_card_image_info import BacklogCardImageInfo

_DEFAULT_TEMPLATES_PATH = "web_interation/templates"
Coordinates = Tuple[int, int]
Shape = Tuple[int, int, int]
Image = cv2.typing.MatLike


class UserStoryImageInfo:
    def __init__(self, color, loyalty, customers, position) -> None:
        self.color = color
        self.loyalty = loyalty
        self.customers = customers
        self.position = position

    def _equals_to_game_user_story(self, user_story: UserStoryCard):
        card_info = user_story.info
        if abs(self.loyalty - card_info.loyalty) > 1e-4:
            return False
        if abs(self.customers - card_info.customers_to_bring) > 1e-4:
            return False
        return True

    def __eq__(self, value: object) -> bool:
        if isinstance(value, UserStoryCard):
            return self._equals_to_game_user_story(value)
        if not isinstance(value, UserStoryImageInfo):
            return False
        return (
            self.color == value.color
            and self.loyalty == value.loyalty
            and self.customers == value.customers
            and self.position == value.position
        )

    def __repr__(self) -> str:
        return f"UserStoryImageInfo({self.color}, {self.loyalty}, {self.customers}, {self.position})"

class GameImageParser:
    def __init__(self, templates_path=_DEFAULT_TEMPLATES_PATH) -> None:
        self.templates_path = templates_path
        self.black = np.array([0, 0, 0])
        self.white = np.array([255, 255, 255])
        self.templates = self._load_templates()

        self.board_positions = {
            (540, 960, 3): {"y_0": 135, "y_1": 495, "x_0": 715, "x_1": 925},
            (1028, 1920, 3): {"x_0": 1372, "y_0": 268, "x_1": 1750, "y_1": 939},
        }

        self.rows_params = {
            (540, 960, 3): {"w": 88, "h": 37, "x_0": 10, "y_0": 48, "height": 46},
            (1028, 1920, 3): {"w": 170, "h": 70, "x_0": 9, "y_0": 81, "height": 87},
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
                "x_0": 902,
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
        if filename[0] == "k":
            return "000"
        return filename[0]

    def _load_templates(self):
        templates: List[Tuple[str, cv2.typing.MatLike]] = []
        for template_filename in listdir(self.templates_path):
            image_char = self._get_image_char(template_filename)
            image = cv2.imread(path.join(self.templates_path, template_filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            templates.append((image_char, image))
        return templates

    def read_digit(self, image: cv2.typing.MatLike, tolerance: float = 0.05):
        best_diff = image.size
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

        if best_diff < tolerance * image.size:
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

    def get_rows(
        self,
        board_image: cv2.typing.MatLike,
        original_shape: Tuple[int, int, int],
        board_position: Coordinates,
    ):
        row_params = self.rows_params[original_shape]
        board_x0, board_y0 = board_position
        rows: List[Tuple[cv2.typing.MatLike, Coordinates]] = []
        w = row_params["w"]
        h = row_params["h"]
        for i in range(6):
            x_0 = row_params["x_0"]
            y_0 = row_params["y_0"] + row_params["height"] * i
            row = board_image[y_0 : y_0 + h, x_0 : x_0 + w]

            color = row[0, 0]
            if (color == [255, 255, 255]).all():
                break

            center_x = board_x0 + x_0 + w // 2
            center_y = board_y0 + y_0 + h // 2
            rows.append((row, (center_x, center_y)))

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
        customers = float(customers) / 1000

        return color, loyalty, customers

    def read_user_stories(self, game_image: cv2.typing.MatLike):
        board, board_position = self.get_board(game_image)
        rows = self.get_rows(board, game_image.shape, board_position)

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
        position: Tuple[int, int],
        original_shape: Tuple[int, int, int],
    ):
        card_params = self.cards_params[original_shape]
        l = card_params["l"]
        r = card_params["r"]
        left: Image = row[:, :l]
        right: Image = row[:, r:]
        if (right[0, 0] == [255, 255, 255]).all():
            return ([left, position],)
        x, y = position
        right_pos = (x + r, y)
        return [left, position], [right, right_pos]

    def get_backlog_card_images(self, image: cv2.typing.MatLike):
        backlog_board, board_position = self.get_board(image)

        backlog_rows = self.get_rows(backlog_board, image.shape, board_position)
        cards = []
        for row, position in backlog_rows:
            row_cards = self.split_row(row, position, image.shape)
            cards.extend(row_cards)

        return cards

    def read_backlog_card_descripton(
        self,
        card_image: cv2.typing.MatLike,
        position: Tuple[int, int],
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

        x, y = position
        height, width, *_ = card_image.shape
        position = (x + width // 2, y + height // 2)

        return BacklogCardImageInfo(color, hours_value, position)

    def read_backlog(self, image: cv2.typing.MatLike):
        backlog_cards: List[BacklogCardImageInfo] = []
        cards = self.get_backlog_card_images(image)

        for card, position in cards:
            card_descripton = self.read_backlog_card_descripton(
                card, position, image.shape
            )
            backlog_cards.append(card_descripton)

        return backlog_cards


def load_characters():
    characters = []
    template_dir = getcwd()
    if "web_interaction" not in template_dir:
        template_dir = path.join(template_dir, "web_interaction")
    template_dir = path.join(template_dir, "templates")

    for f in listdir(template_dir):
        key = "" if f[:5] == "empty" else f[0]
        digit = cv2.imread(path.join(template_dir, f))
        characters.append((key, digit))
    return characters


CHARACTERS = load_characters()


def get_black_white_image(
    image: cv2.typing.MatLike, backgruond_color, original_shape: Tuple[int, int, int]
):
    lower = backgruond_color * 0.6
    upper = backgruond_color * 1.01
    if original_shape == (1028, 1920, 3):
        black = np.array([0, 0, 0])
        mask = cv2.inRange(image, black, black)
    else:
        mask = cv2.inRange(image, lower, upper)
    image = image.copy()
    image[mask == 255] = [255, 255, 255]
    image[mask == 0] = [0, 0, 0]

    return image


def is_loading(image: cv2.typing.MatLike):
    black_color = [0, 0, 0]
    uniform_area = image[5:155, 5:155]
    return (uniform_area == black_color).all()


def find_digit(image):
    for key, value in CHARACTERS:
        if value is None:
            continue
        if value.shape != image.shape:
            continue
        if (image == value).all():
            return key


def get_float(nums, num_width, num_count):
    value = ""
    for i in range(num_count):
        num: cv2.typing.MatLike = nums[:, num_width * i : num_width * (i + 1)]
        digit = find_digit(num)
        if digit == "k":
            value = str(float(value) * 1000)
            break
        if digit is None:
            plt.imshow(num)
            plt.show()
            filename = input()
            y, x, _ = num.shape
            filename += f"_{y}x{x}"
            cwd = os.getcwd()
            if "web_interaction" not in cwd:
                cwd = os.path.join(cwd, "web_interaction")
            cv2.imwrite(os.path.join(cwd, "templates", f"{filename}.png"), num)
            global CHARACTERS
            CHARACTERS = load_characters()
            digit = filename[0]
        value += str(digit)
    return float(value)


user_story_num_width = {
    (540, 960, 3): 6,
    (1028, 1920, 3): 11,
}


def get_user_story_float(
    nums: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
):
    num_width = user_story_num_width[original_shape]
    return get_float(nums, num_width, 6)


backlog_num_width = {
    (540, 960, 3): 11,
    (1028, 1920, 3): 20,
}


def get_backlog_float(nums: cv2.typing.MatLike, original_shape: Tuple[int, int, int]):
    num_width = backlog_num_width[original_shape]
    return get_float(nums, num_width, 2)


loyalty_nums_positions = {
    (540, 960, 3): {"x_0": 49, "y_0": 7, "y_1": 15},
    (1028, 1920, 3): {"x_0": 95, "y_0": 15, "y_1": 28},
}


def get_user_story_loyalty(
    user_story: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
):
    position = loyalty_nums_positions[original_shape]
    x_0 = position["x_0"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    loyalty_nums = user_story[y_0:y_1, x_0:]
    loyalty_value = get_user_story_float(loyalty_nums, original_shape)
    return loyalty_value


customers_nums_positions = {
    (540, 960, 3): {"x_0": 49, "y_0": 19, "y_1": 27},
    (1028, 1920, 3): {"x_0": 95, "y_0": 36, "y_1": 49},
}


def get_user_story_customers(
    user_story: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
):
    position = customers_nums_positions[original_shape]
    x_0 = position["x_0"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    customers_nums = user_story[y_0:y_1, x_0:]
    customers_value = get_user_story_float(customers_nums, original_shape)
    return customers_value / 1000


def get_user_story_description(
    user_story: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
):
    color = np.array(user_story[0, 0])
    user_story_bw = get_black_white_image(user_story, color, original_shape)

    loyalty_value = get_user_story_loyalty(user_story_bw, original_shape)
    customers_value = get_user_story_customers(user_story_bw, original_shape)

    color = frozenset(enumerate(color))

    return color, loyalty_value, customers_value


board_positions = {
    (540, 960, 3): {"y_0": 135, "y_1": 495, "x_0": 715, "x_1": 925},
    (1028, 1920, 3): {"x_0": 1463, "y_0": 268, "x_1": 1841, "y_1": 939},
    # (1028, 1920, 3): {"x_0": 1372, "y_0": 268, "x_1": 1750, "y_1": 939},
}


def get_board(image: cv2.typing.MatLike):
    position = board_positions[image.shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    board = image[y_0:y_1, x_0:x_1]
    return board


rows_params = {
    (540, 960, 3): {"w": 88, "h": 37, "x_0": 10, "y_0": 48, "height": 46},
    (1028, 1920, 3): {"w": 169, "h": 70, "x_0": 9, "y_0": 81, "height": 88},
}


def get_rows(board_image: cv2.typing.MatLike, origingal_shape: Tuple[int, int, int]):
    row_params = rows_params[origingal_shape]
    position = board_positions[origingal_shape]
    board_x0 = position["x_0"]
    board_y0 = position["y_0"]
    rows = []
    w = row_params["w"]
    h = row_params["h"]
    for i in range(6):
        x_0 = row_params["x_0"]
        y_0 = row_params["y_0"] + row_params["height"] * i
        row = board_image[y_0 : y_0 + h, x_0 : x_0 + w]
        # plt.imshow(row)
        # plt.show()

        color = row[0, 0]
        if (color == [255, 255, 255]).all():
            break

        rows.append((row, (board_x0 + x_0, board_y0 + y_0)))

    return rows


def get_user_stories(frame: cv2.typing.MatLike):
    user_stories = []
    positions = []
    user_stories_board = get_board(frame)
    user_stories_cards = get_rows(user_stories_board, frame.shape)
    for user_story, position in user_stories_cards:
        description = get_user_story_description(user_story, frame.shape)
        user_stories.append(description)
        positions.append(position)

    return user_stories, positions


cards_params = {
    (540, 960, 3): {"l": 42, "r": 46},
    (1028, 1920, 3): {"l": 81, "r": 87},
}


def split_row(
    row: cv2.typing.MatLike,
    position: Tuple[int, int],
    original_shape: Tuple[int, int, int],
):
    card_params = cards_params[original_shape]
    l = card_params["l"]
    r = card_params["r"]
    left = row[:, :l]
    right = row[:, r:]
    if (right[0, 0] == [255, 255, 255]).all():
        return [left], [position]
    x, y = position
    right_pos = (x + r, y)
    return [left, right], [position, right_pos]


hours_positions = {
    (540, 960, 3): {"x_0": 3, "x_1": 25, "y_0": 9, "y_1": 24},
    (1028, 1920, 3): {"x_0": 8, "x_1": 48, "y_0": 17, "y_1": 44},
}


def get_backlog_card_descripton(
    card_image: cv2.typing.MatLike,
    position: Tuple[int, int],
    original_shape: Tuple[int, int, int],
):
    color = np.array(card_image[0, 0])
    card_image = get_black_white_image(card_image, color, original_shape)

    hours_position = hours_positions[original_shape]
    x_0 = hours_position["x_0"]
    x_1 = hours_position["x_1"]
    y_0 = hours_position["y_0"]
    y_1 = hours_position["y_1"]

    hours = card_image[y_0:y_1, x_0:x_1]

    hours_value = get_backlog_float(hours, original_shape)

    color = frozenset(enumerate(color))

    return color, hours_value, position


def get_backlog_card_images(image: cv2.typing.MatLike):
    backlog_board = get_board(image)

    backlog_rows = get_rows(backlog_board, image.shape)
    cards = []
    positions = []
    for row, position in backlog_rows:
        row_cards, row_positions = split_row(row, position, image.shape)
        cards.extend(row_cards)
        positions.extend(row_positions)

    return cards, positions


def get_backlog(image: cv2.typing.MatLike):
    backlog_cards = []
    cards, positions = get_backlog_card_images(image)

    for card, position in zip(cards, positions):
        card_descripton = get_backlog_card_descripton(card, position, image.shape)
        backlog_cards.append(card_descripton)

    return backlog_cards


sprint_positions = {
    (540, 960, 3): {"y_0": 14, "y_1": 30, "x_0": 487, "x_1": 630, "width": 11},
    (1028, 1920, 3): {"x_0": 902, "y_0": 7, "x_1": 1100, "y_1": 32, "width": 21},
}


def get_sprint_number(
    meta_info: cv2.typing.MatLike, original_shape: Tuple[int, int, int]
):
    position = sprint_positions[original_shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    width = position["width"]

    sprint = meta_info[y_0:y_1, x_0:x_1]
    sprint_n = get_float(sprint, width, 5)
    return sprint_n


money_positions = {
    (540, 960, 3): {"y_0": 33, "y_1": 49, "x_0": 421, "x_1": 480, "width": 11},
    (1028, 1920, 3): {"x_0": 750, "y_0": 44, "x_1": 900, "y_1": 69, "width": 21},
}


def get_game_money(meta_info: cv2.typing.MatLike, original_shape: Tuple[int, int, int]):
    position = money_positions[original_shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    width = position["width"]
    money = meta_info[y_0:y_1, x_0:x_1]
    unique_colors = np.unique(money[:, 0], axis=0)
    while len(unique_colors) == 1:
        money = money[:, 1:]
        unique_colors = np.unique(money[:, 0], axis=0)

    money_value = get_float(money, width, 5)
    return money_value


customers_positions = {
    (540, 960, 3): {"y_0": 18, "y_1": 29, "x_0": 161, "x_1": 206, "width": 9},
    (1028, 1920, 3): {"x_0": 291, "y_0": 12, "x_1": 900, "y_1": 34, "width": 18},
}


def get_customers(meta_info: cv2.typing.MatLike, original_shape: Tuple[int, int, int]):
    position = customers_positions[original_shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    num_width = position["width"]
    num_count = 6
    image_width = num_width * num_count
    customers_nums = meta_info[y_0:y_1, x_0 : x_0 + image_width]

    customers_value = get_float(customers_nums, num_width, num_count)
    return customers_value / 1000


loyalty_positions = {
    (540, 960, 3): {"y_0": 38, "y_1": 49, "x_0": 143, "x_1": 206, "width": 9},
    (1028, 1920, 3): {"x_0": 255, "y_0": 49, "x_1": 900, "y_1": 71, "width": 18},
}


def get_loyalty(meta_info: cv2.typing.MatLike, original_shape: Tuple[int, int, int]):
    position = loyalty_positions[original_shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    num_width = position["width"]
    num_count = 6
    image_width = num_width * num_count
    loyalty_nums = meta_info[y_0:y_1, x_0 : x_0 + image_width]

    loyalty_value = get_float(loyalty_nums, num_width, num_count)
    return loyalty_value


def get_current_sprint_hours(backlog_image):
    backlog_board = get_board(backlog_image)
    button = backlog_board[334:356, 11:199]

    button_action = button[:, :100]
    button_action_digit = find_digit(button_action)
    if button_action_digit == "d":
        nums = button[7:15, 115:145]
    else:
        nums = button[8:16, 138:168]
    nums = get_black_white_image(nums, nums[0, 0])
    current_hours_nums = nums[:, :12]
    current_hours_value = get_float(current_hours_nums, 6, 2)

    return current_hours_value


meta_info_positions = {
    (540, 960, 3): {"x_0": 57, "y_0": 7, "x_1": 932, "y_1": 83},
    (1028, 1920, 3): {"x_0": 184, "y_0": 36, "x_1": 1794, "y_1": 136},
}


def get_meta_info_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    position = meta_info_positions[image.shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    return image[y_0:y_1, x_0:x_1]


def main():
    # image = cv2.imread("web_interaction/game_state.png")
    image = cv2.imread("tests/test_images/backlog_1028_1980.png")
    # image = cv2.imread("tests/test_images/user_stories_1028_1980.png")
    # image = cv2.imread("web_interaction\game_state1.png")
    print(image.shape)
    original_shape = image.shape
    meta_info = get_meta_info_image(image)

    sprint_n = get_sprint_number(meta_info, original_shape)
    print(sprint_n)

    money = get_game_money(meta_info, original_shape)
    print(money)

    customers_value = get_customers(meta_info, original_shape)
    print(customers_value)

    loyalty_value = get_loyalty(meta_info, original_shape)
    print(loyalty_value)

    # current_sprint_hours = get_current_sprint_hours(image)
    # print(current_sprint_hours)

    image = cv2.imread("tests/test_images/user_stories_1028_1980.png")
    user_stories = get_user_stories(image)
    print(user_stories)

    image = cv2.imread("tests/test_images/backlog_1028_1980.png")
    backlog_cards = get_backlog(image)
    print(backlog_cards)


if __name__ == "__main__":
    main()
