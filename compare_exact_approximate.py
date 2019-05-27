import numpy as np
from typing import *
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.lines import Line2D
import os
from mintracker import ExactMinTracker, ApproximateMinTracker


def create_random_points(num: int):
    """
    Generates num random datapoints between 0 and 1
    """
    datavalues = np.random.rand(num, 3)
    return datavalues


def make_plots(discarded_set: np.ndarray,
               non_minimal_candidates: np.ndarray,
               minimal_set: np.ndarray,
               step: int,
               version: str):
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

    file_path = os.getcwd() + '/3Dplots/' + version
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_name = file_path + '/figure0' + str(step).zfill(2) + '.png'
    plt.savefig(file_name)
    plt.tight_layout(pad=2)
    #plt.show()
    plt.close()


def main():
    # set up data and slack vector
    dataset = create_random_points(1000)
    slack = [0.2, 0.2, 0.2]  # slack variable to set by decision maker
    epsilon = [0.0001, 0.0001, 0.0001]

    mintracker_exact = ExactMinTracker(slack)
    mintracker_approximate = ApproximateMinTracker(slack, epsilon)

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 3))
        mintracker_exact.update_mintracker(x)
        mintracker_approximate.update_mintracker(x)
        #minimals, non_minimal_candidates = mintracker_exact.get_minimals()
        #minimals_a, non_minimal_candidates_a = mintracker_approximate.get_minimals()

        #make_plots(mintracker_exact.discarded, non_minimal_candidates, minimals, i, 'exact')
        #make_plots(mintracker_approximate.discarded, non_minimal_candidates_a, minimals_a, i, 'approximate')

    print('Exact Solutions:')
    print(mintracker_exact.get_minimals())
    print('Approximate Solutions:')
    print(mintracker_approximate.get_minimals())



if __name__ == "__main__":
    main()
