from typing import *

import numpy as np


class EBOStreamTracker:
    """EBOStreamTracker which keeps track of the candidate set of input elements and those who can be discarded for the
    EBO procedure. An input element is a candidate if it is the current decision or it might be come the decision in
    the future. All inputs are discarded which are not candidates, meaning that they are and never will be the decision
    of the a lexicographic semiorder at hand.
    """

    def __init__(self, slack: List[float]):
        self.slack = slack
        self.candidates = None
        self.discarded = None
        self.dim = len(slack)
        self.number_comparisons = 0
        self.max_candidates = 1

    def digest(self, x: np.ndarray):
        """
        Updates the EBO Tracker when a new input element is available
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

                if current[0] > y[0] + self.slack[0]:
                    return

                if y[0] > current[0] + self.slack[0]:
                    index_filter[j] = False
                    continue

                if all(np.greater_equal(current, y)):
                    return

                if all(np.greater(y, current)):
                    index_filter[j] = False
                    continue

            self.candidates = np.vstack((self.candidates[index_filter, :], x))

        self.max_candidates = self.max_candidates if len(self.candidates) < self.max_candidates else len(self.candidates)


def ebo(data_set: np.ndarray, sigma: List[float]):
    """
    Selects one point from a set X most suitable according to the lexicographic semiorder structure.

    :param data_set: input set X
    :param sigma: threshold parameters
    :return: one single point
    """
    decision = data_set.copy()
    dim = len(sigma)

    # Survivor set method on X
    for i in range(dim):
        if len(decision) == 1:
            return decision[0, :]
        min_value = np.amin(decision[:, i])
        sort = decision[:, i] <= min_value + sigma[i]
        decision = decision[sort, :]

    # lexicographic selection on survivor set
    for i in range(dim):
        if len(decision) == 1:
            return decision[0, :]
        min_value = np.amin(decision[:, i])
        sort = decision[:, i] > min_value
        decision = decision[np.invert(sort), :]

    return decision[0, :]