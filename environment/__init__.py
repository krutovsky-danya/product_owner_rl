from .environment import ProductOwnerEnv
from .credit_payer_env import CreditPayerEnv
from environment.StochasticGameStartEnv import StochasticGameStartEnv


def make(is_vanilla=True):
    if is_vanilla:
        return ProductOwnerEnv()
    return StochasticGameStartEnv()
