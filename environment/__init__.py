from .environment import ProductOwnerEnv
from .credit_payer_env import CreditPayerEnv
from .tutorial_solver_env import TutorialSolverEnv
from environment.stochastic_game_start_env import StochasticGameStartEnv


def make(is_vanilla=True):
    if is_vanilla:
        return ProductOwnerEnv()
    return StochasticGameStartEnv()
