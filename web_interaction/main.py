from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time


def open_game():
    driver = webdriver.Chrome()

    driver.get("https://npg-team.itch.io/product-owner-simulator")

    load_iframe_btn = driver.find_element(by=By.CLASS_NAME, value="load_iframe_btn")
    load_iframe_btn.click()

    return driver


def start_game(driver):
    pass


def main():
    driver = open_game()

    time.sleep(10)

    fullscreen_btn = driver.find_element(by=By.CLASS_NAME, value="fullscreen_btn")
    fullscreen_btn.click()
    time.sleep(1)

    iframe = driver.find_element(by=By.ID, value="game_drop")

    height = iframe.rect["height"]  # 540
    width = iframe.rect["width"]  # 960
    print(iframe.rect)

    # skip intro
    iframe.click()
    iframe.click()

    time.sleep(1)

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

    driver.quit()


if __name__ == "__main__":
    main()
