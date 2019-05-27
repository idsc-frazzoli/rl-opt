import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from optimization.mintracker import ExactMinTracker, ApproximateMinTracker


def create_random_points(num: int):
    """
    Generates num random datapoints between 0 and 1
    """
    datavalues = np.random.rand(num, 3)
    return datavalues


def main():
    # set up data and slack vector
    dataset = create_random_points(1000)
    slack = [0.2, 0.2, 0.2]  # slack variable to set by decision maker
    epsilon = [0.0001, 0.0001, 0.0001]

    mintracker_exact = ExactMinTracker(slack)
    mintracker_approximate = ApproximateMinTracker(slack, epsilon)

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 3))
        mintracker_exact.update_mintracker(x)
        mintracker_approximate.update_mintracker(x)
        minimals, non_minimal_candidates = mintracker_exact.get_minimals()
        minimals_a, non_minimal_candidates_a = mintracker_approximate.get_minimals()

    print('Exact Solutions:')
    print(mintracker_exact.get_minimals())
    print('Approximate Solutions:')
    print(mintracker_approximate.get_minimals())



if __name__ == "__main__":
    main()
