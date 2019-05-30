from typing import *

import numpy as np


class PureMinTracker:
    """MinTracker which keeps track of the candidate set of input elements and those who can be discarded.
    An input element is a candidate if it is minimal or it might be come minimal in the future.
    All inputs are discarded which are not candidates, meaning that they are and never will be minimal in a lexicographical
    semiordered sense."""

    def __init__(self, slack: List[float]):
        self.slack = slack
        self.candidates = None
        self.dim = len(slack)

    def get_minimals(self):
        """
        Divides the candidate set into to subsets,
        where one contains the points which are minimal
        and the other is the complement of it.
        """
        try:
            minimals = self.candidates.copy()
        except AttributeError:
            print('Candidate set is empty!')
            return []

        for i in range(self.dim):
            min_value = np.amin(minimals[:, i])
            sort = minimals[:, i] <= min_value + self.slack[i]
            set_sorted = minimals[sort, :]

            minimals = set_sorted

        return minimals


class ExactPureMinTracker(PureMinTracker):
    """Exact MinTracker of the lexicographic semiorder problem."""

    def __init__(self, slack: List[float]):
        PureMinTracker.__init__(self, slack)

    def update_mintracker_exact(self, x: np.ndarray):
        """
        Updates the MinTracker when a new input element is available
        :param x: current applicant point
        :return:
        """
        x = np.reshape(x, (1, self.dim))
        # if candidate set is empty add x
        if self.candidates is None:
            self.candidates = x
            return
        else:
            current = x[0]
            index_filter = np.ones(len(self.candidates), dtype=bool)
            for j in range(len(self.candidates)):
                y = self.candidates[j, :]
                for i in range(0, self.dim):
                    if current[i] > y[i] + self.slack[i]:
                        if self.discardable_exact(y, current, i):
                            return
                    if current[i] + self.slack[i] < y[i]:
                        if self.discardable_exact(current, y, i):
                            index_filter[j] = False
                            break

            self.candidates = np.vstack((self.candidates[index_filter, :], x))

    def discardable_exact(self, x, y, i):
        """

        :param x: Point that possibly discards y
        :param y: Point to be discarded
        :param i: Dimension where strict preference was detected
        :return: True if to be discarded
        """
        for index in range(i):
            if np.greater(x[index], y[index]):
                return False
        return True


class ApproximatePureMinTracker(PureMinTracker):
    """Approximate MinTracker of the lexicographic semiorder problem, where solutions will only be stored if there is
    not yet another point stored which is in in its neighbouhood, e.g. the hypercuboid with edges epsilon
    centered around the point."""

    epsilon: List[float]  # if difference between points in dimension i is less than epsilon_i they are close

    def __init__(self, slack: List[float], epsilon: List[float]):
        PureMinTracker.__init__(self, slack)
        if len(slack) != len(epsilon):
            raise ValueError('Epsilon and slack vector not of same size!')
        self.epsilon = epsilon

    def update_mintracker_approx(self, x: np.ndarray):
        """
        Updates the MinTracker when a new input element is available
        :param x: current applicant point
        :return:
        """
        x = np.reshape(x, (1, self.dim))
        # if candidate set is empty add x
        if self.candidates is None:
            self.candidates = x
            return
        else:
            current = x[0]
            for j in range(len(self.candidates)):
                y = self.candidates[j, :]
                if self.preanalysis(current, y):
                    return

            index_filter = np.ones(len(self.candidates), dtype=bool)
            for j in range(len(self.candidates)):
                y = self.candidates[j, :]
                for i in range(0, self.dim):
                    if current[i] > y[i] + self.slack[i]:
                        if self.discardable_approx(y, current, i):
                            return
                    if current[i] + self.slack[i] < y[i]:
                        if self.discardable_approx(current, y, i):
                            index_filter[j] = False
                            break

            self.candidates = np.vstack((self.candidates[index_filter, :], x))

    def preanalysis(self, x: np.ndarray, y: np.ndarray):
        """
        Checks whether two points are in the same neighbourhood, or if applicant has no advantage in any dimension
        e.g. if they are contained within the same hypercuboid.
        :param x: current applicant point
        :param y: a point already tracked
        :return: True if in the same neighbourhood False otherwise
        """
        lower_bound = np.subtract(y, self.epsilon)
        return not any(np.less(x, lower_bound))

    def discardable_approx(self, x, y, i):
        """

        :param x: Point that possibly discards y
        :param y: Point to be discarded
        :param i: Dimension where strict preference was detected
        :return: True if to be discarded
        """
        for index in range(i):
            if np.greater_equal(x[index] - self.epsilon[index], y[index]):
                return False
        return True
