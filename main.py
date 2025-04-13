import time

from environment.environments_factory import EnvironmentFactory

from pipeline.study_agent import load_dqn_agent

from web_interaction import GameImageParser, GameCoordinator
from web_interaction.web_controller import WebController

from pet_logging import get_logger


def main():
    logger = get_logger("WebInteraction")

    image_parser = GameImageParser("web_interaction/templates")
    game_coordinator = GameCoordinator(image_parser)

    env = EnvironmentFactory().create_credit_env()

    web_controller = WebController(env.game, game_coordinator, logger)

    agent = load_dqn_agent("models/credit_start_model.pt")

    driver = web_controller.play_game(env, agent)

    time.sleep(10)

    driver.quit()


if __name__ == "__main__":
    main()
