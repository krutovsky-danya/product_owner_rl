from game.game import ProductOwnerGame


def get_buggy_game_1():
    game = ProductOwnerGame()
    game.context.current_sprint = 35
    game.context.credit = 0
    game.context.set_money(200_000)
    game.context.set_loyalty(4.5)
    game.context.customers = 40
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)
    return game


def get_buggy_game_2():
    game = ProductOwnerGame()
    game.context.current_sprint = 35
    game.context.credit = 0
    game.context.set_money(77_000 + 50_000)
    game.context.set_loyalty(4.52)
    game.context.customers = 48.08
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)
    game.buy_robot(0)
    return game


def get_buggy_game_3():
    game = ProductOwnerGame()
    game.context.current_sprint = 35
    game.context.credit = 0
    game.context.set_money(161_000)
    game.context.set_loyalty(4.42)
    game.context.customers = 46.54
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)
    return game


def get_game_on_sprint_26():
    game = ProductOwnerGame()
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)
    game.buy_robot(0)
    game.context.current_sprint = 26
    game.context.credit = 75_000
    game.context.set_money(34_000)
    game.context.set_loyalty(4.27)
    game.context.customers = 37.03
    return game


def get_game_on_sprint_21():
    game = ProductOwnerGame()
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)
    game.context.current_sprint = 21
    game.context.credit = 120_000
    game.context.set_money(75_000)
    game.context.set_loyalty(4.21)
    game.context.customers = 35.42
    return game


def get_game_on_sprint_6():
    game = ProductOwnerGame()
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)
    game.buy_robot(0)
    game.context.current_sprint = 6
    game.context.credit = 255_000
    game.context.set_money(28_000)
    game.context.set_loyalty(4.03)
    game.context.customers = 27.53
    return game
