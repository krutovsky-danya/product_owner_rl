from environment.environment import ProductOwnerEnv
from game.backlog_card.backlog_card import Card
from game.backlog_card.card_info import CardInfo
from game.game import ProductOwnerGame
from game.userstory_card.userstory_card import UserStoryCard
from game.userstory_card.userstory_card_info import UserStoryCardInfo
from itertools import groupby
from operator import itemgetter
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from .single_color_storage import SingleColorStorage
from typing import List, Sequence, Tuple, FrozenSet

import cv2
import time
from . import image_parser


def open_game() -> WebDriver:
    driver = webdriver.Chrome()

    driver.get("https://npg-team.itch.io/product-owner-simulator")

    load_iframe_btn = driver.find_element(by=By.CLASS_NAME, value="load_iframe_btn")
    load_iframe_btn.click()

    return driver


def wait_loading(iframe: WebElement):
    while True:
        time.sleep(1)
        iframe.screenshot("game_state.png")
        loading_image = cv2.imread("game_state.png")
        is_loading = image_parser.is_loading(loading_image)
        if not is_loading:
            break


def start_game(driver, iframe: WebElement):
    # skip intro
    iframe.click()
    iframe.click()

    height = iframe.rect["height"]  # 540
    width = iframe.rect["width"]  # 960

    # type name
    ActionChains(driver).move_to_element_with_offset(
        iframe, 0, int(0.1 * height)
    ).click().send_keys("selenium").perform()

    # start game
    ActionChains(driver).move_to_element_with_offset(
        iframe, 0, int(0.2 * height)
    ).click().perform()

    # skip tutorial
    ActionChains(driver).move_to_element_with_offset(
        iframe, -int(0.35 * width), int(0.4 * height)
    ).click().perform()

    time.sleep(1)

    # turn off sprint animation
    ActionChains(driver).move_to_element_with_offset(
        iframe, -int(0.45 * width), -int(0.42 * height)  # move to settings icon
    ).click().move_to_element_with_offset(
        iframe, int(0.1 * width), int(0.07 * height)  # move to animation checkbox
    ).click().move_to_element_with_offset(
        iframe, -int(0.45 * width), -int(0.42 * height)  # move to settings icon
    ).click().perform()


def select_user_story_board(driver, iframe: WebElement, width: int, height: int):
    position = board_icons_positions[(height, width)]
    x = position["x_off"]
    y = position["user_stories_y"]
    click_on_element(driver, iframe, x, y)
    time.sleep(2)


def select_backlog_board(driver, iframe: WebElement, width: int, height: int):
    position = board_icons_positions[(height, width)]
    x = position["x_off"]
    y = position["backlog_y"]
    click_on_element(driver, iframe, x, y)


board_action_positions = {
    (540, 960): {"x": 817, "y": 480},
    (1028, 1920): {"x": 1654, "y": 911},
}


def click_board_button(driver, iframe: WebElement, width: int, height: int):
    position = board_action_positions[(height, width)]
    x = position["x"]
    y = position["y"]
    click_on_element(driver, iframe, x, y)


def click_on_element(driver, iframe: WebElement, x: int, y: int):
    height = iframe.rect["height"]  # 540
    width = iframe.rect["width"]  # 960

    x_offset = x - width // 2 + 1
    y_offset = y - height // 2 + 1

    ActionChains(driver).move_to_element_with_offset(
        iframe, x_offset, y_offset
    ).click().perform()


def click_user_story(driver, iframe: WebElement, x: int, y: int):
    height = iframe.rect["height"]  # 540
    width = iframe.rect["width"]  # 960

    select_user_story_board(driver, iframe, width, height)
    click_on_element(driver, iframe, x, y)


def buy_research(driver, iframe: WebElement, width: int, height: int):
    select_user_story_board(driver, iframe, width, height)
    ActionChains(driver).move_to_element_with_offset(
        iframe, int(0.3 * width), -int(0.3 * height)
    ).click().perform()


def get_game_user_stories_from_image(image: cv2.typing.MatLike, current_sprint: int):
    game_user_stories: List[UserStoryCardInfo] = []
    user_stories, positions = image_parser.get_user_stories(image)

    for user_story, position in zip(user_stories, positions):
        color, loyalty, customers = user_story
        color_storage = SingleColorStorage(color)
        game_user_story = UserStoryCardInfo("S", current_sprint, color_storage)
        game_user_story.loyalty = loyalty
        game_user_story.customers_to_bring = customers
        game_user_story.position = position
        game_user_stories.append(game_user_story)

    return game_user_stories


def insert_user_stories_from_image(game: ProductOwnerGame, image: cv2.typing.MatLike):
    user_stories = get_game_user_stories_from_image(image, game.context.current_sprint)
    game.userstories.stories_list.clear()
    game.context.available_stories.clear()
    for user_story in user_stories:
        game.userstories.add_us(user_story)


def insert_backlog_cards_from_image(game: ProductOwnerGame, image: cv2.typing.MatLike):
    backlog_cards = image_parser.get_backlog(image)

    backlog_cards_by_color = {}
    for key, group in groupby(backlog_cards, itemgetter(0)):
        backlog_cards_by_color[key] = list(group)

    for user_story in game.userstories.release:
        info = user_story.info
        us_id_val = id(info)
        info.related_cards.clear()
        color = info.color
        backlog_cards = backlog_cards_by_color[color]

        print(color)
        for _, hours, position in backlog_cards:
            print(hours)
            card_info = CardInfo(
                hours_val=hours,
                color_val=color,
                us_id_val=us_id_val,
                label_val=info.label,
                card_type_val=info.card_type,
            )
            card_info.position = position
            info.related_cards.append(card_info)


def fill_game_main_info_from_image(game: ProductOwnerGame, image: cv2.typing.MatLike):
    context = game.context
    meta_info = image_parser.get_meta_info_image(image)

    # current_sprint_hours = image_parser.get_current_sprint_hours(image)
    # context.current_sprint_hours = current_sprint_hours

    current_sprint = image_parser.get_sprint_number(meta_info, image.shape)
    context.current_sprint = current_sprint

    money = image_parser.get_game_money(meta_info, image.shape)
    context.set_money(money)

    loyalty = image_parser.get_loyalty(meta_info, image.shape)
    context.set_loyalty(loyalty)

    context.customers = image_parser.get_customers(meta_info, image.shape)

    credit = max(300_000 - (current_sprint - 1) * 9_000, 0)
    context.credit = credit


board_icons_positions = {
    (540, 960): {"x_on": 700, "x_off": 950, "backlog_y": 245, "user_stories_y": 396},
    (1028, 1920): {
        "x_on": 1434,
        "x_off": 1892,
        "backlog_y": 466,
        "user_stories_y": 754,
    },
}


def apply_start_sprint_action(
    driver, iframe: WebElement, width: int, height: int, env: ProductOwnerEnv
):
    print("Start new sprint")

    select_backlog_board(driver, iframe, width, height)
    time.sleep(2)

    click_board_button(driver, iframe, width, height)
    time.sleep(2)

    if env.game.context.current_sprint == 34:
        ActionChains(driver).move_to_element(iframe).click().perform()
        time.sleep(2)

    env._perform_start_sprint_action()

    filename = "game_state.png"
    iframe.screenshot(filename)
    game_image = cv2.imread(filename)
    # os.remove(filename)

    fill_game_main_info_from_image(env.game, game_image)

    # position = board_icons_positions[(height, width)]
    # x = position["x_on"]
    # backlog_y = position["backlog_y"]
    # user_stories_y = position["user_stories_y"]
    # click_on_element(driver, iframe, x, backlog_y)
    # click_on_element(driver, iframe, x, user_stories_y)


def apply_decompose_action(
    driver, iframe: WebElement, width: int, height: int, env: ProductOwnerEnv
):
    print("Start decomposition")
    select_user_story_board(driver, iframe, width, height)
    click_board_button(driver, iframe, width, height)
    time.sleep(1)

    filename = "game_state.png"
    iframe.screenshot(filename)
    image = cv2.imread(filename)
    # os.remove(filename)

    insert_backlog_cards_from_image(env.game, image)

    env._perform_decomposition()


def apply_user_story_action(
    action: int, driver, iframe: WebElement, env: ProductOwnerEnv
):
    print("Start user story action:", action)
    user_story = env.userstory_env.get_encoded_card(action)
    print("User story:", user_story)

    click_user_story(driver, iframe, *user_story.info.position)

    reward = env._perform_action_userstory(action)

    filename = "game_state.png"
    iframe.screenshot(filename)
    image = cv2.imread(filename)
    # os.remove(filename)

    insert_user_stories_from_image(env.game, image)

    print(reward)


def find_backlog_card_position(card: Card, image: cv2.typing.MatLike):
    card: CardInfo = card.info
    backlog_cards = image_parser.get_backlog(image)
    for color, hours, position in backlog_cards:
        if color != card.color:
            continue
        if hours != card.hours:
            continue
        return position


def apply_backlog_card_action(
    action: int, driver, iframe: WebElement, env: ProductOwnerEnv
):
    height = iframe.rect["height"]  # 540
    width = iframe.rect["width"]  # 960
    print("Start moving backlog card")
    select_backlog_board(driver, iframe, width, height)

    time.sleep(2)

    filename = "game_state.png"
    iframe.screenshot(filename)
    backlog_board_image = cv2.imread(filename)
    # os.remove(filename)

    card = env.backlog_env.get_card(action)
    print("Selected card", card)
    position = find_backlog_card_position(card, backlog_board_image)
    print("Found at position", position)

    click_on_element(driver, iframe, *position)

    print("Clicked on card")

    env._perform_action_backlog_card(action)


release_button_positions = {
    (540, 960): {"x": 115, "y": 455},
    (1028, 1920): {"x": 211, "y": 865},
}


def apply_release_action(
    driver, iframe: WebElement, width: int, height: int, env: ProductOwnerEnv
):
    position = release_button_positions[(height, width)]
    x = position["x"]
    y = position["y"]
    click_on_element(driver, iframe, x, y)

    time.sleep(1)

    filename = "game_state.png"
    iframe.screenshot(filename)
    game_image = cv2.imread(filename)
    # os.remove(filename)

    env._perform_release()
    fill_game_main_info_from_image(env.game, game_image)


def apply_buy_statistical_research_action(
    driver, iframe: WebElement, env: ProductOwnerEnv
):
    height = iframe.rect["height"]  # 540
    width = iframe.rect["width"]  # 960
    buy_research(driver, iframe, width, height)

    env._perform_statistical_research()

    filename = "game_state.png"
    iframe.screenshot(filename)
    userstory_image = cv2.imread(filename)
    # os.remove(filename)

    insert_user_stories_from_image(env.game, userstory_image)


def main():
    driver = open_game()

    iframe = driver.find_element(by=By.ID, value="game_drop")

    wait_loading(iframe)

    start_game(driver, iframe)

    time.sleep(10)

    driver.quit()


if __name__ == "__main__":
    main()
