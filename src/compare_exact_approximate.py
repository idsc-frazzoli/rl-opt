import numpy as np

from optimization.helpers import find_equals, not_contained_in_exact, not_contained_in_approximate, maximal_difference
from optimization.mintracker import ExactMinTracker, ApproximateMinTracker


def create_random_points(num: int):
    """
    Generates num random datapoints between 0 and 1
    """
    datavalues = np.random.rand(num, 3)
    return datavalues


def main():
    # set up data and slack vector
    dataset = create_random_points(10000)
    slack = [0.2, 0.2, 0.2]  # slack variable to set by decision maker
    epsilon = [0.1, 0.1, 0.1]

    mintracker_exact = ExactMinTracker(slack)
    mintracker_approximate = ApproximateMinTracker(slack, epsilon)

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 3))
        mintracker_exact.update_mintracker_exact(x)
        mintracker_approximate.update_mintracker_approx(x)
        minimals, non_minimal_candidates = mintracker_exact.get_minimals()
        minimals_a, non_minimal_candidates_a = mintracker_approximate.get_minimals()

    print('Exact Solutions:')
    print(minimals)
    print(f'Number of exact solutions: {len(minimals)}')
    print('Approximate Solutions:')
    print(minimals_a)
    print(f'Number of approximate solutions: {len(minimals_a)}')

    find_equals(minimals_a, minimals)
    different_entries = not_contained_in_exact(minimals_a, minimals)
    maximal_difference(different_entries, minimals)
    not_contained_in_approximate(minimals_a, minimals)


if __name__ == "__main__":
    main()
