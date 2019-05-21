import numpy as np
from typing import *


class MinTracker:
    """MinTracker which keeps track of the candidate set of input elements and those who can be discarded.
    An input element is a candidate if it is minimal or it might be come minimal in the future.
    All inputs are discarded which are not candidates, meaning that they are and never will be minimal in a lexicographical
    semiordered sense."""

    def __init__(self, slack: List[float]):
        self.slack = slack
        self.candidates = None
        self.discarded = None
        self.dim = len(slack)

    def update_mintracker(self, x: np.ndarray):
        """Updates the MinTracker when a new input element is available"""
        # if candidate set is empty add x
        if self.candidates is None:
            self.candidates = x
            return
        else:
            index_filter = np.ones(len(self.candidates), dtype=bool)
            for j in range(len(self.candidates)):
                y = self.candidates[j, :]
                current = x[0]
                for i in range(0, self.dim):
                    if current[i] > y[i] + self.slack[i]:
                        self.update_discarded(x)
                        return
                    if current[i] + self.slack[i] < y[i]:
                        discardable = True
                        for index in range(i):
                            discardable = discardable and np.less_equal(current[index], y[index])
                        if discardable:
                            index_filter[j] = False
                            break

            for item in self.candidates[np.invert(index_filter), :]:
                self.update_discarded(item)

            self.candidates = np.vstack((self.candidates[index_filter, :], x))

    def __update_discarded(self, x):
        """Adds an element to the discarded set"""
        if self.discarded is None:
            self.discarded = np.reshape(x, (1, self.dim))
            return
        self.discarded = np.vstack((self.discarded, x))

    def get_minimals(self):
        """
        Divides the set into to subsets,
        where one contains the points which are minimal
        and the is the complement of it.
        """
        minimals = self.candidates.copy()
        set_retained = []

        for i in range(self.dim):
            min = np.amin(minimals[:, i])
            sort = minimals[:, i] <= min + self.slack[i]
            set_sorted = minimals[sort, :]
            try:
                set_retained = np.concatenate((set_retained, minimals[np.invert(sort), :]), axis=0)
            except ValueError:
                set_retained = minimals[np.invert(sort), :]

            minimals = set_sorted

        return minimals, set_retained
