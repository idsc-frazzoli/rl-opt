import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
import os


# import data from csv as np.array
def getData():
    """
    Retrieves the input data from the csv files.
    """
    file_name = "inputData3D.csv"
    df = pd.read_csv(file_name)
    return df.values


def getRetainerSet(set):
    """
    Divides the set into to subsets,
    where one conatins the points which have to be retained for future analysis
    and the other contains the discard points

    """
    set_sorted = set[np.lexsort(np.rot90(set))]
    x_min = set_sorted[0, 0]
    x_filter = set_sorted[:, 0] <= x_min + slack[0]
    set_filtered_x = set_sorted[x_filter, :]
    discarded_x = set_sorted[np.invert(x_filter), :]
    y_filter = []
    y_discarded = []
    y_min = set_filtered_x[0, 1]
    for i in range(len(set_filtered_x)):
        y_min = y_min if y_min < set_filtered_x[i, 1] else set_filtered_x[i, 1]
        if set_filtered_x[i, 1] <= y_min + slack[1]:
            y_filter.append(i)
        else:
            y_discarded.append(i)
    set_sorted_y = set_filtered_x[y_filter, :]
    discarded_y = np.concatenate((discarded_x, set_filtered_x[y_discarded, :]), axis=0)
    z_sort = []
    z_discarded = []
    z_min = set_filtered_x[0, 2]
    for i in range(len(set_sorted_y)):
        z_min = z_min if z_min < set_sorted_y[i, 2] else set_sorted_y[i, 2]
        if set_sorted_y[i, 2] <= z_min + slack[2]:
            z_sort.append(i)
        else:
            z_discarded.append(i)
    return set_sorted_y[z_sort, :], np.concatenate((discarded_y, set_sorted_y[z_discarded, :]), axis=0)


def getMinElements(set):
    """
    Divides the set into to subsets,
    where one contains the points which are minimal
    and the is the complement of it.
    """
    x_min = np.amin(set[:, 0])
    x_sort = set[:, 0] <= x_min + slack[1]
    set_sorted_x = set[x_sort, :]
    set_retained_x = set[np.invert(x_sort), :]

    y_min = np.amin(set_sorted_x[:, 1])
    y_sort = set_sorted_x[:, 1] <= y_min + slack[1]
    set_sorted_y = set_sorted_x[y_sort, :]
    set_retained_y = np.concatenate((set_retained_x, set_sorted_x[np.invert(y_sort), :]), axis=0)

    z_min = np.amin(set[:, 2])
    z_sort = set_sorted_y[:, 2] <= z_min + slack[2]
    return set_sorted_y[z_sort, :], np.concatenate((set_retained_y, set_sorted_y[np.invert(z_sort), :]), axis=0)


def makePlots(setInput, setRetainer, setMin):
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
    # generate scatter plots of the data
    ax.scatter(setInput[:, 0], setInput[:, 1], setInput[:, 2], marker=".", zorder=1, label='Input Elements')
    ax.scatter(setRetainer[:, 0], setRetainer[:, 1], setRetainer[:, 2], marker=".", zorder=2, label='Retained Elements')
    ax.scatter(setMin[:, 0], setMin[:, 1], setMin[:, 2], marker=".", zorder=3, label='Minimal Elements')
    plt.legend(loc='upper left')
    file_path = os.getcwd() + "/3Dplots/figure0" + str(i).zfill(2) + ".png"
    plt.savefig(file_path)
    plt.show()


# set up data and slack vector
dataset = getData()
slack = np.array([2, 2, 2])  # slack variable to set by decision maker

# iterate through data inputs and divide into disjoint sets: Discarded points, retained points and minimal points
for i in np.arange(len(dataset)):
    currentInputSet = dataset[0:i + 1]
    feasibleSet, discardedSet = getRetainerSet(currentInputSet)
    minimalElements, retainedSet = getMinElements(feasibleSet)
    makePlots(discardedSet, retainedSet, minimalElements)
