from typing import *

import numpy as np


class MinTracker:
    """MinTracker which keeps track of the candidate set of input elements and those who can be discarded.
    An input element is a candidate if it is minimal or it might be come minimal in the future.
    All inputs are discarded which are not candidates, meaning that they are and never will be minimal in a lexicographical
    semiordered sense.

    This class shall be primarily used for evaluation and animation purposes.
    """

    def __init__(self, slack: List[float]):
        self.slack = slack
        self.candidates = None
        self.discarded = None
        self.dim = len(slack)
        self.number_comparisons = 0
        self.max_candidates = 0

    def get_minimals(self):
        """
        Divides the candidate set into to subsets,
        where one contains the points which are minimal
        and the other is the complement of it.
        :return: minimals, non minimal candidates
        """
        try:
            minimals = self.candidates.copy()
        except AttributeError:
            print('Candidate set is empty!')
            return [], []

        set_retained = []

        for i in range(self.dim):
            min_value = np.amin(minimals[:, i])
            sort = minimals[:, i] <= min_value + self.slack[i]
            set_sorted = minimals[sort, :]
            try:
                set_retained = np.concatenate((set_retained, minimals[np.invert(sort), :]), axis=0)
            except ValueError:
                set_retained = minimals[np.invert(sort), :]

            minimals = set_sorted

        return minimals, set_retained

    def update_discarded(self, x):
        """Adds an element to the discarded set"""
        if self.discarded is None:
            self.discarded = np.reshape(x, (1, self.dim))
            return
        self.discarded = np.vstack((self.discarded, x))

    def get_number_of_comparisons(self):
        return self.number_comparisons


class ExactMinTracker(MinTracker):
    """Exact MinTracker of the lexicographic semiorder problem."""

    def __init__(self, slack: List[float]):
        MinTracker.__init__(self, slack)

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
                self.number_comparisons += 1
                y = self.candidates[j, :]
                for i in range(0, self.dim):
                    if current[i] > y[i] + self.slack[i]:
                        if self.discardable_exact(y, current, i):
                            self.update_discarded(x)
                            return
                    if current[i] + self.slack[i] < y[i]:
                        if self.discardable_exact(current, y, i):
                            index_filter[j] = False
                            break

            for item in self.candidates[np.invert(index_filter), :]:
                self.update_discarded(item)

            self.candidates = np.vstack((self.candidates[index_filter, :], x))

        self.max_candidates = self.max_candidates if len(self.candidates) < self.max_candidates else len(self.candidates)

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