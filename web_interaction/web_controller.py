import cv2
import logging
import time

from selenium.webdriver import ActionChains
from selenium.webdriver.remote.webelement import WebElement

from environment.environment import ProductOwnerEnv
from web_interaction import GameCoordinator


class WebController:
    def __init__(self, game_coordinator: GameCoordinator, logger: logging.Logger):
        self.game_coordinator = game_coordinator
        self.logger = logger
        self.board_icons_positions = {
            (540, 960): {
                "x_on": 700,
                "x_off": 950,
                "backlog_y": 245,
                "user_stories_y": 396,
            },
            (1028, 1920): {
                "x_on": 1434,
                "x_off": 1892,
                "backlog_y": 466,
                "user_stories_y": 754,
            },
        }

        self.board_action_positions = {
            (540, 960): {"x": 817, "y": 480},
            (1028, 1920): {"x": 1654, "y": 911},
        }

    def click_on_element(self, driver, iframe: WebElement, x: int, y: int):
        height = iframe.rect["height"]
        width = iframe.rect["width"]

        x_offset = x - width // 2 + 1
        y_offset = y - height // 2 + 1

        ActionChains(driver).move_to_element_with_offset(
            iframe, x_offset, y_offset
        ).click().perform()

    def click_board_button(self, driver, iframe: WebElement):
        height = iframe.rect["height"]
        width = iframe.rect["width"]
        position = self.board_action_positions[(height, width)]
        x = position["x"]
        y = position["y"]
        self.click_on_element(driver, iframe, x, y)

    def select_user_story_board(self, driver, iframe: WebElement):
        height = iframe.rect["height"]
        width = iframe.rect["width"]
        position = self.board_icons_positions[(height, width)]
        x = position["x_off"]
        y = position["user_stories_y"]
        self.click_on_element(driver, iframe, x, y)
        time.sleep(1)

    def select_backlog_board(self, driver, iframe: WebElement):
        height = iframe.rect["height"]
        width = iframe.rect["width"]
        position = self.board_icons_positions[(height, width)]
        x = position["x_off"]
        y = position["backlog_y"]
        self.click_on_element(driver, iframe, x, y)
        time.sleep(1)

    def click_user_story(self, driver, iframe: WebElement, x: int, y: int):
        self.select_user_story_board(driver, iframe)
        self.click_on_element(driver, iframe, x, y)

    def apply_user_story_action(
        self, action: int, driver, iframe: WebElement, env: ProductOwnerEnv
    ):
        self.logger.info(f"Start user story action: {action}")
        user_story = env.userstory_env.get_encoded_card(action)
        self.logger.info(f"User story: {user_story}")

        position = self.game_coordinator.find_user_story_position(user_story)
        self.logger.info(f"Found at position: {position}")

        self.click_user_story(driver, iframe, *position)

        reward = env._perform_action_userstory(action)

        filename = "game_state.png"
        iframe.screenshot(filename)
        image = cv2.imread(filename)
        # os.remove(filename)

        self.game_coordinator.insert_user_stories_from_image(env.game, image)

        self.logger.info(f"Reward: {reward}")

    def apply_decompose_action(self, driver, iframe: WebElement, env: ProductOwnerEnv):
        self.logger.info("Start decomposition")
        self.select_user_story_board(driver, iframe)
        self.click_board_button(driver, iframe)
        time.sleep(1)

        filename = "game_state.png"
        iframe.screenshot(filename)
        image = cv2.imread(filename)
        # os.remove(filename)

        self.game_coordinator.insert_backlog_cards_from_image(env.game, image)

        env._perform_decomposition()

    def apply_backlog_card_action(
        self, action: int, driver, iframe: WebElement, env: ProductOwnerEnv
    ):
        self.logger.info("Start moving backlog card")
        self.select_backlog_board(driver, iframe)

        card = env.backlog_env.get_card(action)
        self.logger.info(f"Selected card {card}")

        position = self.game_coordinator.find_backlog_card_position(card.info)
        self.logger.info(f"Found at position {position}")

        self.click_on_element(driver, iframe, *position)
        self.logger.info("Clicked on card")

        self.game_coordinator.remove_backlog_card_from_backlog(card.info)

        env._perform_action_backlog_card(action)

    def start_sprint(
        self, driver, iframe: WebElement, env: ProductOwnerEnv
    ):
        self.logger.info("Start new sprint")

        self.select_backlog_board(driver, iframe)
        time.sleep(1)

        self.click_board_button(driver, iframe)
        time.sleep(1)

        if env.game.context.current_sprint == 34:
            ActionChains(driver).move_to_element(iframe).click().perform()
            time.sleep(1)

        env._perform_start_sprint_action()

        filename = "game_state.png"
        iframe.screenshot(filename)
        game_image = cv2.imread(filename)
        # os.remove(filename)

        self.game_coordinator.update_header_info(env.game, game_image)
