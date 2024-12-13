import cv2
import logging
import os
import time

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from environment.environment import ProductOwnerEnv, ProductOwnerGame
from .game_coordinator import GameCoordinator


class WebController:
    def __init__(
        self,
        game: ProductOwnerGame,
        game_coordinator: GameCoordinator,
        logger: logging.Logger,
    ):
        self.game = game
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

        self.release_button_positions = {
            (540, 960): {"x": 115, "y": 455},
            (1028, 1920): {"x": 211, "y": 865},
        }

        self.robot_positions = {(1028, 1920): {"x": 1100, "y": 930}}

        self.screenshot_count = 0

    def take_screenshot(self, iframe: WebElement):
        os.makedirs("game_states", exist_ok=True)
        filename = f"game_states/{self.screenshot_count:03d}.png"
        iframe.screenshot(filename)
        self.screenshot_count += 1
        image = cv2.imread(filename)
        # os.remove(filename)

        return image

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

        self.game_coordinator.clear_backlog_sprint(self.game)
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

        game_image = self.take_screenshot(iframe)

        self.game_coordinator.user_stories = (
            self.game_coordinator.image_parser.read_user_stories(game_image)
        )

        self.logger.info(f"Reward: {reward}")

    def decompose(self, driver, iframe: WebElement, env: ProductOwnerEnv):
        self.logger.info("Start decomposition")
        self.select_user_story_board(driver, iframe)
        self.click_board_button(driver, iframe)
        time.sleep(1)

        game_image = self.take_screenshot(iframe)

        self.game_coordinator.insert_backlog_cards_from_image(env.game, game_image)

        self.logger.info(
            f"Backlog cards appered: {self.game_coordinator.backlog_cards}"
        )

        env._perform_decomposition()

    def apply_backlog_card_action(
        self, action: int, driver, iframe: WebElement, env: ProductOwnerEnv
    ):
        self.logger.info("Start moving backlog card")
        self.select_backlog_board(driver, iframe)

        card = env.backlog_env.get_card(action)
        self.logger.info(f"Selected card {card}")

        screenshot = self.take_screenshot(iframe)
        self.game_coordinator.backlog_cards = (
            self.game_coordinator.image_parser.read_backlog(screenshot)
        )

        position = self.game_coordinator.find_backlog_card_position(card.info)
        self.logger.info(f"Found at position {position}")

        self.click_on_element(driver, iframe, *position)
        self.logger.info("Clicked on card")

        self.game_coordinator.remove_backlog_card_from_backlog(card.info)

        env._perform_action_backlog_card(action)

    def start_sprint(self, driver, iframe: WebElement, env: ProductOwnerEnv):
        self.logger.info("Start new sprint")

        self.select_backlog_board(driver, iframe)
        time.sleep(1)

        self.click_board_button(driver, iframe)
        time.sleep(1)

        if env.game.context.current_sprint == 34:
            ActionChains(driver).move_to_element(iframe).click().perform()
            time.sleep(1)

        env._perform_start_sprint_action()

        if env.game.context.done:
            return

        game_image = self.take_screenshot(iframe)

        self.game_coordinator.update_header_info(env.game, game_image)

    def release_tasks(self, driver, iframe: WebElement, env: ProductOwnerEnv):
        height = iframe.rect["height"]
        width = iframe.rect["width"]
        self.logger.info("Release tasks")
        position = self.release_button_positions[(height, width)]
        x = position["x"]
        y = position["y"]
        self.click_on_element(driver, iframe, x, y)
        time.sleep(1)

        game_image = self.take_screenshot(iframe)

        env._perform_release()
        self.game_coordinator.update_header_info(env.game, game_image)

    def buy_robot(self, driver, iframe: WebElement, env: ProductOwnerEnv):
        self.logger.info("Buy robot")

        height = iframe.rect["height"]
        width = iframe.rect["width"]
        buy_robot_position = self.robot_positions[(height, width)]
        self.click_on_element(driver, iframe, **buy_robot_position)

        env._perform_buy_robot()

    def buy_research(self, driver, iframe: WebElement):
        height = iframe.rect["height"]
        width = iframe.rect["width"]
        self.select_user_story_board(driver, iframe)
        ActionChains(driver).move_to_element_with_offset(
            iframe, int(0.3 * width), -int(0.3 * height)
        ).click().perform()

    def buy_statistical_research(
        self, driver, iframe: WebElement, env: ProductOwnerEnv
    ):
        self.logger.info("Buy statistical research")
        self.buy_research(driver, iframe)

        env._perform_statistical_research()

        userstory_image = self.take_screenshot(iframe)

        self.game_coordinator.insert_user_stories_from_image(env.game, userstory_image)

        self.logger.info(f"User stories appeared: {self.game_coordinator.user_stories}")

    def open_game(
        self,
        url: str = "https://npg-team.itch.io/product-owner-simulator",
    ) -> WebDriver:
        driver = webdriver.Chrome()

        driver.get(url)

        load_iframe_btn = driver.find_element(by=By.CLASS_NAME, value="load_iframe_btn")
        load_iframe_btn.click()

        return driver

    def find_game(self, driver: WebDriver):
        iframe = driver.find_element(by=By.ID, value="game_drop")
        return iframe

    def wait_loading(self, iframe: WebElement):
        while True:
            loading_image = self.take_screenshot(iframe)
            is_loading = self.game_coordinator.image_parser.is_loading(loading_image)
            if not is_loading:
                break
            time.sleep(1)

    def open_full_sceen(self, driver: WebDriver):
        fullscreen_button = driver.find_element(
            by=By.CLASS_NAME, value="fullscreen_btn"
        )
        fullscreen_button.click()

    def start_game(self, driver: WebDriver, iframe: WebElement):
        # skip intro
        iframe.click()
        iframe.click()

        height = iframe.rect["height"]  # 540
        width = iframe.rect["width"]  # 960

        # type name
        ActionChains(driver).move_to_element_with_offset(
            iframe, 0, int(0.1 * height)
        ).click().send_keys("DDQN").perform()

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

    def apply_web_action(
        self, action: int, driver, iframe: WebElement, env: ProductOwnerEnv
    ):
        if action == 0:  # start sprint
            self.start_sprint(driver, iframe, env)
            return

        if action == 1:  # decompose
            self.decompose(driver, iframe, env)
            return

        if action == 2:  # release
            self.release_tasks(driver, iframe, env)
            return

        if action == 3:
            self.buy_robot(driver, iframe, env)
            return

        if action == 5:  # buy statistical research
            self.buy_statistical_research(driver, iframe, env)
            return

        if action >= env.meta_action_dim:
            action -= env.meta_action_dim
        else:
            raise Exception(f"Acton not handled: {action}")

        if action < env.userstory_env.max_action_num:
            self.apply_user_story_action(action, driver, iframe, env)
            return

        action -= env.userstory_env.max_action_num

        if action < env.backlog_env.backlog_max_action_num:
            self.apply_backlog_card_action(action, driver, iframe, env)
            return

        raise Exception(f"Unknown action: {action}")

    def play_game(
        self,
        env: ProductOwnerEnv,
        agent,
        url: str = "https://krutovsky-danya.itch.io/productownersimulator",
    ) -> WebDriver:
        driver = self.open_game(url)
        iframe = self.find_game(driver)
        self.open_full_sceen(driver)
        self.wait_loading(iframe)
        self.start_game(driver, iframe)
        image = self.take_screenshot(iframe)

        self.game_coordinator.skip_tutorial(env.game)
        self.game_coordinator.insert_user_stories_from_image(env.game, image)
        self.game_coordinator.update_header_info(env.game, image)
        self.game_coordinator.log_game_state(env.game, self.logger)

        self.play_credit_stage(env, agent, driver, iframe)
        self.play_free_stage(env, driver, iframe)

        return driver

    def play_credit_stage(self, env: ProductOwnerEnv, agent, driver, iframe):
        while not env.game.context.done:
            self.game_coordinator.log_game_state(env.game, self.logger)
            state = env.recalculate_state()

            info = env.get_info()

            action = agent.get_action(state, info)
            self.logger.info(f"Action id: {action}")

            self.apply_web_action(action, driver, iframe, env)

            if env.game.context.current_sprint >= 35:
                self.logger.warning("Reached credit end!")
                break

    def play_free_stage(self, env: ProductOwnerEnv, driver, iframe):
        while not env.game.context.done:
            self.apply_web_action(0, driver, iframe, env)
