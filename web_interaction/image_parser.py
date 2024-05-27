import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, getcwd, path
from typing import Tuple

BOARD_X0 = 715  # 1372
BOARD_X1 = 925  # 1750
BOARD_Y0 = 135  # 268
BOARD_Y1 = 495  # 939


class UserStoryImageInfo:
    def __init__(self, color, loyalty, customers, position) -> None:
        self.color = color
        self.loyalty = loyalty
        self.customers = customers
        self.position = position


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


def get_black_white_image(image: cv2.typing.MatLike, backgruond_color):
    lower = backgruond_color * 0.6
    upper = backgruond_color * 1.01
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
            filename += f'_{y}x{x}'
            cv2.imwrite(f"web_interaction/templates/{filename}.png", num)
            global CHARACTERS
            CHARACTERS = load_characters()
            digit = filename[0]
        value += str(digit)
    return float(value)


def get_user_story_float(nums):
    num_width = 6
    return get_float(nums, num_width, 5)


def get_backlog_float(nums):
    num_width = 11
    return get_float(nums, num_width, 2)


def get_user_story_loyalty(user_story):
    loyalty_nums = user_story[7:15, 55:]
    loyalty_value = get_user_story_float(loyalty_nums)
    return loyalty_value


def get_user_story_customers(user_story):
    customers_nums = user_story[19:27, 55:]
    customers_value = get_user_story_float(customers_nums)
    return customers_value / 1000


def get_user_story_description(user_story):
    color = np.array(user_story[0, 0])
    user_story_bw = get_black_white_image(user_story, color)

    loyalty_value = get_user_story_loyalty(user_story_bw)
    customers_value = get_user_story_customers(user_story_bw)

    color = frozenset(enumerate(color))

    return color, loyalty_value, customers_value


def get_board(image: cv2.typing.MatLike):
    board = image[BOARD_Y0:BOARD_Y1, BOARD_X0:BOARD_X1]
    return board


def get_rows(board_image: cv2.typing.MatLike):
    rows = []
    w, h = 88, 37
    for i in range(10):
        x_0 = 10
        y_0 = 48 + 46 * i
        row = board_image[y_0 : y_0 + h, x_0 : x_0 + w]

        color = row[0, 0]
        if (color == [255, 255, 255]).all():
            break

        rows.append((row, (BOARD_X0 + x_0, BOARD_Y0 + y_0)))

    return rows


def get_user_stories(frame):
    user_stories = []
    positions = []
    user_stories_board = get_board(frame)
    user_stories_cards = get_rows(user_stories_board)
    for user_story, position in user_stories_cards:
        description = get_user_story_description(user_story)
        user_stories.append(description)
        positions.append(position)

    return user_stories, positions


def split_row(row: cv2.typing.MatLike, position: Tuple[int, int]):
    left = row[:, :42]
    right = row[:, 46:]
    if (right[0, 0] == [255, 255, 255]).all():
        return [left], [position]
    x, y = position
    right_pos = (x + 46, y)
    return [left, right], [position, right_pos]


def get_backlog_card_descripton(
    card_image: cv2.typing.MatLike, position: Tuple[int, int]
):
    color = np.array(card_image[0, 0])
    card_image = get_black_white_image(card_image, color)

    hours = card_image[9:24, 3:25]
    hours_value = get_backlog_float(hours)

    color = frozenset(enumerate(color))

    return color, hours_value, position


def get_backlog_card_images(image):
    backlog_board = get_board(image)

    backlog_rows = get_rows(backlog_board)
    cards = []
    positions = []
    for row, position in backlog_rows:
        row_cards, row_positions = split_row(row, position)
        cards.extend(row_cards)
        positions.extend(row_positions)

    return cards, positions


def get_backlog(image):
    backlog_cards = []
    cards, positions = get_backlog_card_images(image)

    for card, position in zip(cards, positions):
        card_descripton = get_backlog_card_descripton(card, position)
        backlog_cards.append(card_descripton)

    return backlog_cards

sprint_positions = {
    (540, 960, 3): {'y_0': 14, 'y_1': 30, 'x_0': 487, 'x_1': 530, 'width': 11},
    (1028, 1920, 3): {"x_0": 902, "y_0": 7, "x_1": 1000, "y_1": 32, 'width': 21},
}

def get_sprint_number(meta_info: cv2.typing.MatLike, original_shape: Tuple[int, int, int]):
    position = sprint_positions[original_shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    width = position['width']

    sprint = meta_info[y_0:y_1, x_0:x_1]
    sprint_n = get_float(sprint, width, 3)
    return sprint_n

money_positions = {
    (540, 960, 3): {'y_0': 33, 'y_1': 49, 'x_0': 421, 'x_1': 480, 'width': 11},
    (1028, 1920, 3): {"x_0": 750, "y_0": 44, "x_1": 900, "y_1": 69, 'width': 21},
}

def get_game_money(meta_info: cv2.typing.MatLike, original_shape: Tuple[int, int, int]):
    position = money_positions[original_shape]
    x_0 = position["x_0"]
    x_1 = position["x_1"]
    y_0 = position["y_0"]
    y_1 = position["y_1"]
    width = position['width']
    money = meta_info[y_0:y_1, x_0:x_1]
    unique_colors = np.unique(money[:, 0], axis=0)
    while len(unique_colors) == 1:
        money = money[:, 1:]
        unique_colors = np.unique(money[:, 0], axis=0)

    money_value = get_float(money, width, 5)
    return money_value


def get_customers(meta_info: cv2.typing.MatLike):
    num_width = 9
    num_count = 6
    image_width = num_width * num_count
    customers_nums = meta_info[18:29, 161 : 161 + image_width]

    customers_value = get_float(customers_nums, num_width, num_count)
    return customers_value / 1000


def get_loyalty(meta_info: cv2.typing.MatLike):
    loyalty_nums = meta_info[38:49, 143:206]

    loyalty_value = get_float(loyalty_nums, 9, 5)
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
    image = cv2.imread('tests/test_images/yellow_backlog.png')
    original_shape = image.shape
    meta_info = get_meta_info_image(image)

    sprint_n = get_sprint_number(meta_info, original_shape)
    print(sprint_n)

    money = get_game_money(meta_info, original_shape)
    print(money)

    customers_value = get_customers(meta_info)
    print(customers_value)

    loyalty_value = get_loyalty(meta_info)
    print(loyalty_value)

    # current_sprint_hours = get_current_sprint_hours(image)
    # print(current_sprint_hours)

    # user_stories = get_user_stories(image)
    # print(user_stories)

    backlog_cards = get_backlog(image)
    print(backlog_cards)


if __name__ == "__main__":
    main()
