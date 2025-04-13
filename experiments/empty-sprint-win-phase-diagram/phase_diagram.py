import sys

sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from game import ProductOwnerGame


def prepare_game(game: ProductOwnerGame):
    game.context.is_new_game = False
    game.is_first_release = False
    game.userstories.disable_restrictions()
    game.office.toggle_purchases(True)

    game.context.current_sprint = 35
    game.context.credit = 0

    return game


def get_win_sprint(game: ProductOwnerGame):
    while not game.context.done and game.context.customers > 0:
        game.backlog_start_sprint()

    if game.context.customers <= 0:
        return np.nan

    return game.context.current_sprint


def get_win_sprint_from_initial_state(loyalty, customers):
    game = ProductOwnerGame()
    game = prepare_game(game)

    game.context.set_money(200_000)

    game.context.set_loyalty(loyalty)
    game.context.customers = customers

    return get_win_sprint(game)


def main():
    loyalty_range = np.linspace(1.0, 5.0, 100)
    customers_range = np.linspace(30, 100, 100)

    phase_diagram = np.zeros((len(loyalty_range), len(customers_range)))

    for j, loyalty in enumerate(loyalty_range):
        for i, customers in enumerate(customers_range):
            win_sprint = get_win_sprint_from_initial_state(loyalty, customers)
            phase_diagram[i, j] = win_sprint

    plt.imshow(
        phase_diagram,
        extent=[
            loyalty_range[0],
            loyalty_range[-1],
            customers_range[0],
            customers_range[-1],
        ],
        aspect="auto",
        origin="lower",
        # cmap="gray",
    )
    plt.colorbar()
    plt.ylabel("Customers")
    plt.xlabel("Loyalty")
    plt.title("Win sprint")
    plt.savefig("phase_diagram.png")
    plt.show()


if __name__ == "__main__":
    main()
