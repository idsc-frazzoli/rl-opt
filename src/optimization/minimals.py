from typing import *

import numpy as np


def lex_semiorder_comparison(x: np.ndarray, y: np.ndarray, sigma: List[float]):
    """
    Semiorder lexicographic comparison between two vectors.
    :param x: Vector x
    :param y: Vector y
    :param sigma: threshold parameters
    :return: True if xRy, e.g xPy or xIy
    """
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


def adjacency_matrix(data_set: np.ndarray, sigma: List[float]):
    """
    Computes adjacency matrix (or matrix representation) for R on X.
    :param data_set: in put set X
    :param sigma: threshold parameters
    :return: Adjacency matrix M of <X,R>
    """
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
    """
    Computes adjacency matrix (or matrix representation) for P on X.

    :param a_matrix: Adjacency matrix for R on X
    :return: Adjacency matrix M of <X,P>
    """
    transpose = np.transpose(a_matrix)
    dual = np.logical_not(transpose)

    return np.logical_and(a_matrix, dual)


def warshall(a_strict_matrix: np.ndarray):
    """
    Computes transitive closure of R on X using Warshall algorithm.

    :param a_strict_matrix: Adjacency matrix M of <X,P>
    :return: Adjacency matrix M^T of <X,P^T>
    """
    n = len(a_strict_matrix)

    trans_clos = a_strict_matrix.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                trans_clos[i, j] = trans_clos[i, j] or (trans_clos[i, k] and trans_clos[k, j])

    return trans_clos


def trans_closure(a_strict_matrix: np.ndarray):
    """
    Computes transitive closure of R on X using Warshall algorithm and matrix multiplications.
    :param a_strict_matrix: Adjacency matrix M of <X,P>
    :return: Adjacency matrix M^T of <X,P^T>
    """
    n = len(a_strict_matrix)

    trans_hull = a_strict_matrix.copy()

    for k in range(n):
        new_trans_hull = np.dot(trans_hull, trans_hull)
        if np.array_equal(new_trans_hull, trans_hull):
            return trans_hull
        else:
            trans_hull = np.logical_or(trans_hull, new_trans_hull)

    return trans_hull


def minimal_cycles(data_set: np.ndarray, sigma: List[float]):
    """
    Computes the minimal elements, e.g. bottom cycles, for the lexicographic semiorder on a set X.
    :param data_set: input set X
    :param sigma: threshold parameters
    :return: minimal set
    """
    n = len(data_set)

    a_matrix = adjacency_matrix(data_set, sigma)
    a_matrix_strict = adj_matrix_strict(a_matrix)
    trans_hull = trans_closure(a_matrix_strict)
    trans_hull_strict = adj_matrix_strict(trans_hull)
    index_filter = np.ones(n, dtype=bool)

    for i in range(n):
        if trans_hull_strict[:, i].any():
            index_filter[i] = False

    minimals = data_set[index_filter, :]
    return minimals


def minimal_diamond(data_set: np.ndarray, sigma: List[float]):
    """
    Computes the minimal elements, if they exist, for the lexicographic semiorder on a set X.
    :param data_set: input set X
    :param sigma: threshold parameters
    :return: minimal set
    """
    n = len(data_set)

    a_matrix = adjacency_matrix(data_set, sigma)
    index_filter = np.ones(n, dtype=bool)

    for i in range(n):
        if not a_matrix[i, :].all():
            index_filter[i] = False

    minimals = data_set[index_filter, :]
    return minimals


def minimal_small(data_set: np.ndarray, sigma: List[float]):
    """
    Computes the least element, if it exists, for the lexicographic semiorder on a set X.
    :param data_set: input set X
    :param sigma: threshold parameters
    :return: minimal set
    """
    n = len(data_set)

    a_matrix = adjacency_matrix(data_set, sigma)
    i = np.identity(n, dtype=bool)

    a_matrix_strict = np.logical_or(adj_matrix_strict(a_matrix), i)

    index_filter = np.zeros(n, dtype=bool)

    for i in range(n):
        if a_matrix_strict[i, :].all():
            index_filter[i] = True

    minimals = data_set[index_filter, :]
    return minimals

