import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from show_utils import show_rewards_fitting, show_win_rate


def main():
    sub_name = f"1500"
    show_rewards_fitting(sub_name)
    show_win_rate(sub_name)


if __name__ == "__main__":
    main()
