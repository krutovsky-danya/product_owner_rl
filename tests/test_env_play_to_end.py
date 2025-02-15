from environment.environments_factory import EnvironmentFactory
from game import game_generators


def test_successfull_play():
    # arrange
    env = EnvironmentFactory().create_full_env()
    env.game = game_generators.get_buggy_game_1()

    # act
    result = env._play_blank_sprints_to_end()

    # assert
    assert result, "Action should be successfull"
    assert env.get_done(env.get_info()), "Game should be finished"
    assert env.game.context.is_victory, "Game shold end successfully"


def test_loss_play():
    # arrange
    env = EnvironmentFactory().create_full_env()
    env.game = game_generators.get_buggy_game_1()
    env.game.context.set_money(10000)

    # act
    result = env._play_blank_sprints_to_end()

    # assert
    assert result, "Action should be successfull"
    assert env.get_done(env.get_info()), "Game should be finished"
    assert not env.game.context.is_victory, "Game shold end poorly"


def test_wrong_action():
    # arrange
    env = EnvironmentFactory().create_full_env()

    # act
    result = env._play_blank_sprints_to_end()

    # assert
    assert not result, "Action shold not fit conditions"
