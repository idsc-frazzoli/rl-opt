import numpy as np
from typing import *

def find_equals(approximate: np.ndarray, exact: np.ndarray):
    count = 0
    print('Equal entries for approximate and exact algorithm:')
    for i in range(len(approximate)):
        current = approximate[i, :]
        for j in range(len(exact)):
            to_compare = exact[j, :]
            if all(current == to_compare):
                count += 1
                print(current)
    print(f'Total number of equal entries: {count}')


def not_contained_in_exact(approximate: np.ndarray, exact: np.ndarray):
    count = 0
    print('Entries found in approximate but not in exact algorithm:')
    index_filter = np.zeros(len(approximate), dtype=bool)
    for i in range(len(approximate)):
        current = approximate[i, :]
        any_equals = False
        for j in range(len(exact)):
            to_compare = exact[j, :]
            if all(current == to_compare):
                any_equals = True

        if not any_equals:
            index_filter[i] = True
            count += 1

    different_entries = approximate[index_filter, :]
    print(f'Total number of entries found in approximate that are different to exact entries: {count}')
    return different_entries


def maximal_difference(different_entries, exact):
    max_diff = 0.0
    for i in range(len(different_entries)):
        current = different_entries[i, :]

        for j in range(len(exact)):
            to_compare = exact[j, :]
            differences = np.subtract(current, to_compare)
            max_difference = np.max(np.abs(differences))
            max_diff = max_diff if max_diff > max_difference else max_difference

    print(f'Maximal difference between points is {max_diff}')
    return max_diff


def not_contained_in_approximate(approximate: np.ndarray, exact: np.ndarray):
    count = 0
    print('Entries found in exact but not in approximate algorithm:')
    index_filter = np.zeros(len(exact), dtype=bool)
    for i in range(len(exact)):
        current = exact[i, :]
        any_equals = False
        for j in range(len(approximate)):
            to_compare = approximate[j, :]
            if all(current == to_compare):
                any_equals = True

        if not any_equals:
            index_filter[i] = True
            count += 1

    different_entries = exact[index_filter, :]
    print(f'Total number of entries found in exact that are different to approximate entries: {count}')
    return different_entries
