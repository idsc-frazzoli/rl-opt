import os

import numpy as np
import pandas as pd


def set_subtraction(set_a: np.ndarray, set_b: np.ndarray):
    """

    :param set_a: superset
    :param set_b: subset
    :return: set_a - set_b
    """
    if set_b is None:
        return set_a

    index_filter = np.zeros(len(set_a), dtype=bool)
    for i in range(len(set_a)):
        current = set_a[i, :]
        any_equals = False
        for j in range(len(set_b)):
            to_compare = set_b[j, :]

            if all(current == to_compare):
                any_equals = True

        if not any_equals:
            index_filter[i] = True

    different_entries = set_a[index_filter, :]

    return different_entries


def create_random_points(dim: int, num: int):
    """
    Generates num random data points between 0 and 1
    """
    data_values = np.random.rand(num, dim)
    return data_values


def get_data_from_csv(file_name: str):
    """
    Retrieve Data from csv file and return dataframe
    :return: np.array of data values
    """
    file_name = os.path.join('resources', file_name)
    df = pd.read_csv(file_name)
    return df.values
