import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
from matplotlib.lines import Line2D
import os


class MinTracker:
    """MinTracker"""

    candidates = []
    discarded = []

    def __init__(self, slack):
        self.slack = slack

    def update_mintracker(self, x):
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

            self.candidates = self.candidates[index_filter, :]
            self.candidates = np.vstack((self.candidates, x))

    def update_discarded(self, x):
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


# import data from csv as np.array
def getData():
    """
    Retrieves the input data from the csv files.
    """
    file_name = "inputData3D.csv"
    df = pd.read_csv(file_name)
    return df.values


def createRandomPoints(num):
    """
    Generates num random datapoints between 0 and 20
    """
    datavalues = np.random.rand(num, 3)
    return datavalues * 20


def makePlots(setInput, setRetainer, setMin,i):
    """
    Generates the 3D plots for three sets:
    """
    # set up plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Lexicographic Survivor Set')
    # Setting the axes properties
    ax.set_xlim3d([0.0, 20.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 20.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 20.0])
    ax.set_zlabel('Z')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Minimal Elements',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Candidate Elements',
                              markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Discarded Elements',
                              markerfacecolor='y', markersize=10)]
    # generate scatter plots of the data
    if setInput.size != 0:
        ax.scatter(setInput[:, 0], setInput[:, 1], setInput[:, 2], marker=".", color='y', zorder=1, label='Input Elements')
    if setRetainer.size != 0:
        ax.scatter(setRetainer[:, 0], setRetainer[:, 1], setRetainer[:, 2], marker=".", color='g', zorder=2, label='Retained Elements')
    ax.scatter(setMin[:, 0], setMin[:, 1], setMin[:, 2], marker=".", color='b', zorder=3, label='Minimal Elements')
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3)
    file_path = os.getcwd() + "/3Dplots"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_name = file_path + "/figure0" + str(i).zfill(2) + ".png"
    plt.savefig(file_name)
    plt.tight_layout(pad=2)
    plt.show()


def main():
    # set up data and slack vector
    dataset = getData()
    slack = np.array([2, 2, 2])  # slack variable to set by decision maker

    mintracker = MinTracker(slack)

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :],(1,3))
        mintracker.update_mintracker(x)
        minimals, non_minimal_candidates = mintracker.get_minimals()

        makePlots(np.asarray(mintracker.discarded), np.asarray(non_minimal_candidates), minimals, i)


if __name__ == "__main__":
    main()
