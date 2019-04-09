import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os



# import data from csv as np.array
def getData():
    """
    Retrieve Data from csv file
    """
    file_name = "inputData.csv"
    df = pd.read_csv(file_name)
    return df.values


def getRetainerSet(set):
    """
    Create retainer set
    """
    set_sorted = set[np.lexsort(np.rot90(set))]
    x_min = set_sorted[0, 0]
    x_sort = set_sorted[:, 0] <= x_min + slack[0]
    set_sorted = set_sorted[x_sort, :]
    y_sort = []
    y_min = set_sorted[0, 1]
    for i in range(len(set_sorted)):
        y_min = y_min if y_min < set_sorted[i, 1] else set_sorted[i, 1]
        if set_sorted[i, 1] <= y_min + slack[1]:
            y_sort.append(i)
    return set_sorted[y_sort, :]


def getLowerBound(set):
    """
    Get lower bound of retainer set for creation of rectangle in plot
    """
    y_min = set[0, 1]
    y_sort = [0]
    for i in range(1,len(set)):
        if set[i, 1] < y_min:
            y_sort.append(i)
            y_min = set[i, 1]
    return set[y_sort, :]


def getMinElements(set):
    """
    Get minimal data points
    """
    y_min = np.amin(set[:, 1])
    y_sort = set[:, 1] <= y_min + slack[1]
    return set[y_sort, :]


def makePlots(setInput, setRetainer, setMin):
    """
    Create plots for each new point
    """
    plt.scatter(setRetainer[:, 0], setRetainer[:, 1], marker=".", zorder=10, label='Retained Elements')
    plt.scatter(setInput[:, 0], setInput[:, 1], marker=".", zorder=4, label='Input Elements')
    plt.scatter(setMin[:, 0], setMin[:, 1], marker=".", zorder=20, label='Minimal Elements')
    plt.xlabel('Objective 1 ($\sigma = 2$)')
    plt.ylabel('Objective 2 ($\sigma = 2$)')
    plt.axis([0, 10, 0, 10])
    plt.title('Lexicographic Semiorder MinTracker')
    plt.legend(loc='upper left')

    lowerBound = getLowerBound(retainerSet)
    for j in range(len(lowerBound) - 1):
        diff = lowerBound[j + 1, 0] - lowerBound[j, 0]
        rectangle = plt.Rectangle((lowerBound[j, 0], lowerBound[j, 1]), diff, slack[1], alpha=0.5,
                                  fc="CornflowerBlue", zorder=1)
        plt.gca().add_patch(rectangle)

    diff = lowerBound[0, 0] + slack[0] - lowerBound[-1, 0]
    rectangle = plt.Rectangle((lowerBound[-1, 0], lowerBound[-1, 1]), diff, slack[1], alpha=0.5, fc="CornflowerBlue",
                              zorder=1)
    plt.gca().add_patch(rectangle)
    file_path = os.getcwd() + "/2Dplots"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_name = file_path + "/figure0" + str(i).zfill(2) + ".png"
    plt.savefig(file_name)
    plt.show()


# set up
dataset = getData()
slack = np.array([2, 2])  # slack variable to be set by decision maker

# iterate through data inputs and check whether retainer or not and if minimal
for i in np.arange(len(dataset)):
    data_point = dataset[i]
    currentInputSet = dataset[0:i + 1]
    retainerSet = getRetainerSet(currentInputSet)
    minimalElements = getMinElements(retainerSet)
    makePlots(currentInputSet, retainerSet, minimalElements)
