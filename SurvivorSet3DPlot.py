import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
from matplotlib.lines import Line2D
import os
from mintracker import MinTracker


# import data from csv as np.array
def get_data():
    """
    Retrieves the input data from the csv files.
    """
    file_name = "inputData3D.csv"
    df = pd.read_csv(file_name)
    return df.values


def create_random_points(num):
    """
    Generates num random datapoints between 0 and 1
    """
    datavalues = np.random.rand(num, 3)
    return datavalues


def make_plots(discarded_set, non_minimal_candidates, minimal_set, step):
    """
    Generates the 3D plots for three sets:
    """
    # set up plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Lexicographic Survivor Set')
    # Setting the axes properties
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Minimal Elements',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Candidate Elements',
                              markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Discarded Elements',
                              markerfacecolor='y', markersize=10)]

    # generate scatter plots of the data
    if discarded_set is not None:
        ax.scatter(discarded_set[:, 0], discarded_set[:, 1], discarded_set[:, 2],
                   marker=".", color='y', zorder=1, label='Discarded Elements')

    if non_minimal_candidates.size != 0:
        ax.scatter(non_minimal_candidates[:, 0], non_minimal_candidates[:, 1], non_minimal_candidates[:, 2],
                   marker=".", color='g', zorder=2, label='Candidate Elements')

    ax.scatter(minimal_set[:, 0], minimal_set[:, 1], minimal_set[:, 2],
               marker=".", color='b', zorder=3, label='Minimal Elements')

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3)

    file_path = os.getcwd() + "/3Dplots"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_name = file_path + "/figure0" + str(step).zfill(2) + ".png"
    plt.savefig(file_name)
    plt.tight_layout(pad=2)
    #plt.show()
    plt.close()


def main():
    # set up data and slack vector
    dataset = create_random_points(50)
    slack = np.array([0.05, 0.05, 0.05])  # slack variable to set by decision maker

    mintracker_lex_semi = MinTracker(slack)

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 3))
        mintracker_lex_semi.update_mintracker(x)
        minimals, non_minimal_candidates = mintracker_lex_semi.get_minimals()

        make_plots(mintracker_lex_semi.discarded, non_minimal_candidates, minimals, i)


if __name__ == "__main__":
    main()
