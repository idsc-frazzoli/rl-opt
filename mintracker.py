import numpy as np


class MinTracker:
    """MinTracker which keeps track of the candidate set of input elements and those who can be discarded.
    An input element is a candidate if it is minimal or it might be come minimal in the future.
    All inputs are discarded which are not candidates, meaning that they are and never will be minimal in a lexicographical
    semiordered sense."""

    candidates = []
    discarded = []

    def __init__(self, slack):
        self.slack = slack

    def update_mintracker(self, x):
        """Updates the MinTracker when a new input element is available"""
        # if candidate set is empty add x
        if len(self.candidates) == 0:
            self.candidates = x
            return
        else:
            index_filter = np.ones(len(self.candidates), dtype=bool)
            for j in range(len(self.candidates)):
                y = self.candidates[j, :]
                current = x[0]
                for i in range(0, 3):
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

    def update_discarded(self, x):
        """Adds an element to the discarded set"""
        if len(self.discarded) == 0:
            self.discarded = x
            return
        self.discarded = np.vstack((self.discarded, x))

    def get_minimals(self):
        """
        Divides the set into to subsets,
        where one contains the points which are minimal
        and the is the complement of it.
        """
        x_min = np.amin(self.candidates[:, 0])
        x_sort = self.candidates[:, 0] <= x_min + self.slack[0]
        set_sorted_x = self.candidates[x_sort, :]
        set_retained_x = self.candidates[np.invert(x_sort), :]

        y_min = np.amin(set_sorted_x[:, 1])
        y_sort = set_sorted_x[:, 1] <= y_min + self.slack[1]
        set_sorted_y = set_sorted_x[y_sort, :]
        set_retained_y = np.concatenate((set_retained_x, set_sorted_x[np.invert(y_sort), :]), axis=0)

        z_min = np.amin(set_sorted_y[:, 2])
        z_sort = set_sorted_y[:, 2] <= z_min + self.slack[2]
        return set_sorted_y[z_sort, :], np.concatenate((set_retained_y, set_sorted_y[np.invert(z_sort), :]), axis=0)