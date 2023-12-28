import random

import numpy as np

from typing import Sequence

def clamp(x, minimum, maximum):
    if x < minimum:
        return minimum
    elif x > maximum:
        return maximum
    return x


def stepify(s, step):
    return (s // step) * step


def interpolate(value, table: dict):
    keys = sorted(table.keys())
    first_key = keys[0]
    if value <= first_key:
        return table[first_key]

    last_key = keys[-1]
    if value >= last_key:
        return table[last_key]

    for i in range(1, len(keys)):
        if keys[i - 1] < value <= keys[i]:
            a = table[keys[i - 1]]
            b = table[keys[i]]
            u = a + (value - keys[i - 1]) * (b - a) / (keys[i] - keys[i - 1])
            return u

    return None

def sample_n_or_less(collection, count):
    count = min(count, len(collection))
    return random.sample(collection, count)

def sample_n_or_zero(collection: Sequence, count: int):
    if len(collection) == 0:
        return []
    
    replace = len(collection) < count

    result = np.random.choice(collection, size=count, replace=replace)
    return result