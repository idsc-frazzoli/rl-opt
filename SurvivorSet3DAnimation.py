import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation
import pandas as pd
import os


# import data from csv as np.array
def getData():
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
    x_sort = set_sorted[:, 0] <= x_min + slack[0]
    set_sorted_x = set_sorted[x_sort, :]
    discarded_x = set_sorted[np.invert(x_sort), :]
    y_sort = []
    y_discarded = []
    y_min = set_sorted_x[0, 1]
    for i in range(len(set_sorted_x)):
        y_min = y_min if y_min < set_sorted_x[i, 1] else set_sorted_x[i, 1]
        if set_sorted_x[i, 1] <= y_min + slack[1]:
            y_sort.append(i)
        else:
            y_discarded.append(i)
    set_sorted_y = set_sorted_x[y_sort, :]
    discarded_y = np.concatenate((discarded_x, set_sorted_x[y_discarded, :]), axis=0)
    z_sort = []
    z_discarded = []
    z_min = set_sorted_x[0, 2]
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


def update_graph(num):
    """
    Update the graph with the current sets
    """
    currentInputSet = dataset[0:num+1]
    feasibleSet, discardedSet = getRetainerSet(currentInputSet)
    minimalElements, retainedSet = getMinElements(feasibleSet)
    graph1._offsets3d = (discardedSet[:, 0], discardedSet[:, 1], discardedSet[:, 2])
    graph2._offsets3d = (retainedSet[:, 0], retainedSet[:, 1], retainedSet[:, 2])
    graph3._offsets3d = (minimalElements[:, 0], minimalElements[:, 1], minimalElements[:, 2])
    title.set_text('Lexicographic Survivor Set, Step={}'.format(num))


# set up data and slack vector
dataset = getData()
slack = np.array([2, 2, 2])  # slack variable to set by decision maker
data = dataset[0:1]
feasibleSet, discardedSet = getRetainerSet(data)
minimalElements, retainedSet = getMinElements(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('Lexicographic Survivor Set')
# Setting the axes properties
ax.set_xlim3d([0.0, 20.0])
ax.set_xlabel('X')
ax.set_ylim3d([0.0, 20.0])
ax.set_ylabel('Y')
ax.set_zlim3d([0.0, 20.0])
ax.set_zlabel('Z')

graph1 = ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=".", zorder=1, label='Input Elements')
graph2 = ax.scatter(retainedSet[:, 0], retainedSet[:, 1], retainedSet[:, 2], marker=".", zorder=2, label='Retained Elements')
graph3 = ax.scatter(minimalElements[:, 0], minimalElements[:, 1], minimalElements[:, 2], marker=".", zorder=3, label='Minimal Elements')

ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=500, blit=False)
ax.legend()
plt.show()

file_path = os.getcwd() + "/3Dplots/"
ani.save(file_path + 'animation.gif', writer='imagemagick', fps=60)

