from .environment import ProductOwnerEnv
from .credit_payer_env import CreditPayerEnv
from .tutorial_solver_env import TutorialSolverEnv
from .stochastic_game_start_env import StochasticGameStartEnv


def make(is_vanilla=True, seed=None, card_picker_seed=None):
    if is_vanilla:
        return ProductOwnerEnv(seed=seed, card_picker_seed=card_picker_seed)
    return StochasticGameStartEnv(seed=seed, card_picker_seed=card_picker_seed)
