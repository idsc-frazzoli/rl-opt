from typing import *

import numpy as np


def adjacency_matrix(data_set: np.ndarray, sigma: List[float]):
    n = len(data_set)

    # initialize zero matrix of dim n x n
    a_matrix = np.zeros((n, n), dtype=bool)

    for i in range(n):
        x_i = data_set[i, :]
        for j in range(n):
            x_j = data_set[j, :]
            # compare elements and set entry
            if lex_semiorder_comparison(x_i, x_j, sigma):
                a_matrix[i, j] = True

    return a_matrix


def adj_matrix_strict(a_matrix: np.ndarray):
    transpose = np.transpose(a_matrix)
    dual = np.logical_not(transpose)

    return np.logical_and(a_matrix, dual)


def lex_semiorder_comparison(x: np.ndarray, y: np.ndarray, sigma: List[float]):
    dim = len(sigma)

    for i in range(dim):
        # return True if x strictly precedes y in dim i
        if x[i] + sigma[i] < y[i]:
            return True
        # return False if y strictly precedes x in dim i
        elif y[i] + sigma[i] < x[i]:
            return False

    # return True if no strict precedence, i.d. they are indifferent
    return True


def warshall(a_strict_matrix):
    n = len(a_strict_matrix)

    trans_closure = a_strict_matrix.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                e = trans_closure[i, j]
                b = trans_closure[i, k]
                c = trans_closure[k, j]
                trans_closure[i, j] = trans_closure[i, j] or (trans_closure[i, k] and trans_closure[k, j])

    return trans_closure


def minimal_cycles(data_set: np.ndarray, sigma: List[float]):
    n = len(data_set)

    a_matrix = adjacency_matrix(data_set, sigma)
    a_matrix_strict = adj_matrix_strict(a_matrix)
    trans_closure = warshall(a_matrix_strict)
    trans_closure_strict = adj_matrix_strict(trans_closure)
    index_filter = np.ones(n, dtype=bool)

    for i in range(n):
        if trans_closure_strict[:, i].any():
            index_filter[i] = False

    minimals = data_set[index_filter, :]
    return minimals


def ebo(data_set: np.ndarray, sigma: List[float]):
    decision = data_set.copy()
    dim = len(sigma)

    for i in range(dim):
        if len(decision) == 1:
            return decision[0, :]
        min_value = np.amin(decision[:, i])
        sort = decision[:, i] <= min_value + sigma[i]
        decision = decision[sort, :]

    for i in range(dim):
        if len(decision) == 1:
            return decision[0, :]
        min_value = np.amin(decision[:, i])
        sort = decision[:, i] > min_value
        decision = decision[np.invert(sort), :]

    return decision[0, :]
