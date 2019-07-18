import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from optimization.ebo_stream import EBOStreamTracker, ebo
from optimization.helpers import set_subtraction, create_random_points

from mpl_toolkits.mplot3d import axes3d


# mpl_toolkits.mplot3d import axes3d has to be imported


def make_plots(evaluated_set: np.ndarray, candidates: np.ndarray, slack, step: int):
    """
    Generates the 3D plots for three sets:
    """
    # set up plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Elimination by Objective')
    # Setting the axes properties
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='EBO',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Candidate Elements',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Discarded Elements',
                              markerfacecolor='grey', markersize=10)]

    decision = ebo(candidates, slack)

    only_candidates = set_subtraction(candidates, np.reshape(decision, (1, 3)))

    evaluated_set = set_subtraction(evaluated_set, candidates)

    # generate scatter plots of the data
    if evaluated_set is not None:
        ax.scatter(evaluated_set[:, 0], evaluated_set[:, 1], evaluated_set[:, 2],
                   marker=".", color='grey', zorder=1, label='Discarded Elements')

    if only_candidates.size != 0:
        ax.scatter(only_candidates[:, 0], only_candidates[:, 1], only_candidates[:, 2],
                   marker=".", color='b', zorder=2, label='Candidate Elements')

    ax.scatter(decision[0], decision[1], decision[2],
               marker=".", color='r', zorder=3, label='EBO')

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3)

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', '3Dplots', 'steps')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    file_name = 'figure' + str(step).zfill(4) + '.png'
    file_path = os.path.join(outdir, file_name)

    plt.tight_layout(pad=2)
    plt.savefig(file_path)
    # plt.show()
    plt.close()

    im = Image.open(file_path)

    return im


def main():
    # set up data and slack vector
    dataset = create_random_points(3, 20)
    # dataset = get_data_from_csv('inputData3D.csv')
    slack = [0.2, 0.2, 0.2]  # slack variable to set by decision maker

    ebo_tracker = EBOStreamTracker(slack)

    images = []

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 3))
        ebo_tracker.digest(x)
        candidates = ebo_tracker.candidates

        image = make_plots(dataset[:i + 1, :], candidates, slack, i)
        images.append(image)

    dir_path_e = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', '3Dplots')
    file_name = 'animation.gif'

    images[0].save(os.path.join(dir_path_e, file_name),
                   save_all=True,
                   append_images=images[1:],
                   duration=400,
                   loop=0)


if __name__ == "__main__":
    main()
