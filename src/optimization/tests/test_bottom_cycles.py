import unittest

import numpy as np

from optimization.bottom_cycles import lex_semiorder_comparison, adjacency_matrix, adj_matrix_strict, warshall, \
    minimal_cycles, trans_closure
from optimization.bottom_cycles import ebo


class MinTrackerTest(unittest.TestCase):
    def test_lex_semi_comp(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 2, 3]
        y = [1, 2, 3]
        z = [2, 2, 3]
        w = [2.1, 1.8, 3.1]
        q = [1, 2, 4]
        self.assertTrue(lex_semiorder_comparison(x, y, sigma))
        self.assertTrue(lex_semiorder_comparison(y, x, sigma))
        self.assertTrue(lex_semiorder_comparison(x, z, sigma))
        self.assertFalse(lex_semiorder_comparison(z, x, sigma))
        self.assertTrue(lex_semiorder_comparison(z, w, sigma))
        self.assertTrue(lex_semiorder_comparison(w, z, sigma))
        self.assertTrue(lex_semiorder_comparison(q, z, sigma))
        self.assertTrue(lex_semiorder_comparison(x, q, sigma))

    def test_adjacency_matrix(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 2, 3]
        y = [1, 2, 3]
        z = [1, 2, 3]
        data_set = np.vstack((x, y, z))
        print(data_set)
        a = adjacency_matrix(data_set, sigma)
        self.assertTrue(a.all())

    def test_simple_adj_strict(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 2, 3]
        y = [1, 2, 3]
        z = [1, 2, 3]
        data_set = np.vstack((x, y, z))
        a = adjacency_matrix(data_set, sigma)
        a_strict = adj_matrix_strict(a)
        self.assertFalse(a_strict.all())

    def test_adj_strict(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 2, 3]
        y = [1, 2, 3]
        z = [0, 2, 3]
        data_set = np.vstack((x, y, z))
        a = adjacency_matrix(data_set, sigma)
        a_strict = adj_matrix_strict(a)
        print(a_strict)

    def test_warshall_mult_method(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 3, 3]
        y = [1.1, 2, 3]
        z = [1.2, 1, 3]
        data_set = np.vstack((x, y, z))
        a = adjacency_matrix(data_set, sigma)
        a_strict = adj_matrix_strict(a)
        trans_clos = warshall(a_strict)
        trans_clos2 = trans_closure(a_strict)
        self.assertTrue(trans_clos.all())
        self.assertTrue(np.array_equal(trans_clos, trans_clos2))
        x = [1, 3, 3]
        y = [1.1, 2, 3]
        z = [1.2, 1, 3]
        w = [1.3, 0, 3]
        q = [1.1, 4, 3]
        data_set = np.vstack((x, y, z, w, q))
        a = adjacency_matrix(data_set, sigma)
        a_strict = adj_matrix_strict(a)
        self.assertTrue(np.array_equal(warshall(a_strict), trans_closure(a_strict)))


    def test_minimals(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 3, 3]
        y = [1.1, 2, 3]
        z = [1.2, 1, 3]
        data_set = np.vstack((x, y, z))
        minimals = minimal_cycles(data_set, sigma)
        print(minimals)
        for i in range(3):
            for j in range(3):
                self.assertEqual(data_set[i, j], minimals[i, j])

    def test_minimals_2(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 3, 3]
        y = [1.1, 2, 3]
        z = [1.2, 1, 3]
        w = [1.3, 0, 3]
        q = [1.1, 4, 3]
        data_set = np.vstack((x, y, z, w, q))
        minimals = minimal_cycles(data_set, sigma)
        print(minimals)
        for i in range(3):
            for j in range(3):
                self.assertEqual(data_set[i, j], minimals[i, j])

    def test_ebo(self):
        sigma = [0.1, 0.2, 0.3]
        x = [1, 3, 3]
        y = [1.1, 2, 3]
        z = [1.2, 1, 3]
        w = [1.3, 0, 3]
        q = [1.1, 4, 3]
        data_set = np.vstack((x, y, z, w, q))
        minimals = minimal_cycles(data_set, sigma)
        decision = ebo(data_set, sigma)
        print(minimals)
        print(decision)

