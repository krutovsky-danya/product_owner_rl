from algorithms.knapsack import solve_knapsack


def test_game_case_1():
    tasks = [10, 9, 10, 9]
    capacity = 18
    assert solve_knapsack(tasks, capacity) == [9, 9]


def test_game_case_2():
    tasks = [19, 16, 3]
    capacity = 18
    assert solve_knapsack(tasks, capacity) == [16]


def test_game_case_3():
    tasks = [19, 18, 1]
    capacity = 18
    assert solve_knapsack(tasks, capacity) == [18]


def test_game_case_4():
    tasks = [19, 18, 1]
    capacity = 20
    result = sorted(solve_knapsack(tasks, capacity))
    assert result == [1, 19]


def test_game_case_5():
    tasks = [10, 10]
    capacity = 20
    assert solve_knapsack(tasks, capacity) == [10, 10]


def test_abstract_case_1():
    tasks = [5, 7, 11]
    cpacity = 12
    result = sorted(solve_knapsack(tasks, cpacity))
    assert result == [5, 7]


def test_abscract_case_2():
    tasks = [5, 7, 11]
    cpacity = 10
    result = sorted(solve_knapsack(tasks, cpacity))
    assert result == [7]


def task_abstract_case_3():
    tasks = [5, 9, 13, 15]
    cpacity = 22
    result = sorted(solve_knapsack(tasks, cpacity))
    assert result == [9, 13]
