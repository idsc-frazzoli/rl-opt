import numpy as np


def find_equals(approximate: np.ndarray, exact: np.ndarray):
    """

    :param approximate:
    :param exact:
    :return:
    """
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
    """

    :param approximate:
    :param exact:
    :return:
    """
    count = 0
    # print('Entries found in approximate but not in exact algorithm:')
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
    # print(f'Total number of entries found in approximate that are different to exact entries: {count}')
    return different_entries


def no_differences(approximate: np.ndarray, exact: np.ndarray):
    for i in range(len(approximate)):
        current = approximate[i, :]
        any_equals = False
        for j in range(len(exact)):
            to_compare = exact[j, :]
            if all(current == to_compare):
                any_equals = True

        if not any_equals:
            return False

    return True


def maximal_difference(exact, approximate, epsilon):
    """

    :param approximate: approximate minimal set
    :param exact: exact minimal set
    :return: maximal difference of any element in exact compared to approximate set
    """

    different_entries = not_contained_in_approximate(approximate, exact)

    for i in range(len(different_entries)):
        current = different_entries[i, :]

        for j in range(len(approximate)):
            to_compare = exact[j, :]
            differences = np.subtract(current, to_compare)
            max_difference = np.abs(np.max(differences))

            for j in epsilon:
                if max_difference > epsilon[i]:
                    print(f'Maximal difference between points is {max_difference}')


def not_contained_in_approximate(approximate: np.ndarray, exact: np.ndarray):
    """

    :param approximate: approximate set
    :param exact:
    :return:
    """
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


def not_contained_in(set_a: np.ndarray, set_b: np.ndarray):
    """

    :param set_a: subset
    :param set_b: superset
    :return:
    """
    index_filter = np.zeros(len(set_b), dtype=bool)
    for i in range(len(set_b)):
        current = set_b[i, :]
        any_equals = False
        for j in range(len(set_a)):
            to_compare = set_a[j, :]
            if all(current == to_compare):
                any_equals = True

        if not any_equals:
            index_filter[i] = True

    different_entries = set_b[index_filter, :]

    return different_entries
