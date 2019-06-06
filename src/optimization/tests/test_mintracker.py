import unittest

import numpy as np

from optimization.mintracker import ExactMinTracker, ApproximateMinTracker


class MinTrackerTest(unittest.TestCase):
    def test_simple(self):
        slack = [0.1, 0.2, 0.3]
        mintracker = ExactMinTracker(slack)
        self.assertEqual(mintracker.dim, 3)

    def test_discardable(self):
        slack = [0.1, 0.2]
        mintracker = ExactMinTracker(slack)
        x1 = np.array([1, 1])
        x2 = np.array([1, 1.3])
        x3 = np.array([0.095, 1.3])
        self.assertTrue(mintracker.discardable_exact(x1, x2, 1))
        self.assertFalse(mintracker.discardable_exact(x1, x3, 1))
        self.assertTrue(mintracker.discardable_exact(x3, x2, 1))

    def test_update_exact(self):
        slack = [0.1, 0.2]
        mintracker = ExactMinTracker(slack)
        x1 = np.array([1, 1])
        mintracker.update_mintracker_exact(x1)
        print(mintracker.candidates)
        x2 = np.array([0.98, 1.3])
        mintracker.update_mintracker_exact(x2)
        print(mintracker.candidates)

    def test_update_approximate(self):
        slack = [0.1, 0.2, 0.1]
        epsilon = [0.02, 0.05, 0.02]
        mintracker = ApproximateMinTracker(slack, epsilon)
        x1 = np.array([2, 1, 1.2])
        mintracker.update_mintracker_approx(x1)
        print(mintracker.candidates)
        x2 = np.array([2, 1.1, 1])
        mintracker.update_mintracker_approx(x2)
        x3 = np.array([2.01, 0.99, 1.19])
        mintracker.update_mintracker_approx(x3)
        print(mintracker.candidates)
        x4 = np.array([2.01, 1.01, 1.01])
        mintracker.update_mintracker_approx(x4)
        print(mintracker.candidates)
        size = mintracker.candidates.size
        for i in range(10):
            mintracker.update_mintracker_approx(x2)
        self.assertEqual(mintracker.candidates.size, size)
