import sys
import time

sys.path.insert(0, "..")

from environment.backlog_env import BacklogEnv
from environment.environment import ProductOwnerEnv
from environment.reward_system.base_reward_system import BaseRewardSystem
from environment.userstory_env import UserstoryEnv

from pipeline.study_agent import load_dqn_agent

from web_interaction import GameImageParser, GameCoordinator
from web_interaction.web_controller import WebController

from pet_logging import get_logger


def main():
    logger = get_logger("WebInteraction")

    image_parser = GameImageParser("../web_interaction/templates")
    game_coordinator = GameCoordinator(image_parser)

    userstory_env = UserstoryEnv(4, 0, 0)
    backlog_env = BacklogEnv(12, 0, 0, 0, 0, 0)
    reward_system = BaseRewardSystem(config={})
    env = ProductOwnerEnv(
        userstory_env, backlog_env, with_info=True, reward_system=reward_system
    )

    web_controller = WebController(env.game, game_coordinator, logger)

    agent = load_dqn_agent("../models/credit_start_model.pt")

    driver = web_controller.play_game(env, agent)

    time.sleep(10)

    driver.quit()


if __name__ == "__main__":
    main()
