from typing import List


def solve_knapsack(tasks: List[int], capacity: int) -> List[int]:
    collected_items = [None] * (capacity + 1)
    can_collect = [False] * (capacity + 1)
    can_collect[0] = True

    for task in tasks:
        for t in range(capacity, task - 1, -1):
            if not can_collect[t - task]:
                continue
            can_collect[t] = True

            if collected_items[t] is None:
                collected_items[t] = task

    result = []

    while not can_collect[capacity] and capacity > 0:
        capacity -= 1

    while capacity > 0:
        current = collected_items[capacity]
        if current is None:
            break
        result.append(current)
        capacity -= current

    return result
