import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

from optimization.ebo_stream import EBOStreamTracker, ebo
from optimization.helpers import set_subtraction, create_random_points


def make_plots(evaluated_set, candidates, slack, i):
    """
    Creates plots for each new element digested in EBO tracker.
    :param evaluated_set: already seen elements
    :param candidates: candidate elements of already seen elements
    :param slack: thresholds
    :param i: current step
    :return: image of current step
    """

    # set up plot figure
    fig = plt.figure()
    ax = fig.add_axes([0.18, 0.23, 0.64, 0.64])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Setting the axes properties

    ax.set_xlabel(f'Objective 1 ($\sigma = {slack[0]}$)')
    ax.set_ylabel(f'Objective 2 ($\sigma = {slack[1]}$)')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Current Decision',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Candidate Elements',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Discarded Elements',
                              markerfacecolor='grey', markersize=10)]

    plt.title('EBO (Stream of Data)', pad=10)

    ax.legend(handles=legend_elements, bbox_to_anchor=(0., -0.3, 1., .102), loc=10,
              ncol=3, borderaxespad=0.)

    decision = ebo(candidates, slack)

    discarded_set = set_subtraction(evaluated_set, candidates)

    if len(candidates) > 1:
        only_candidates = set_subtraction(candidates, np.reshape(decision, (1, 2)))
    else:
        only_candidates = []

    if len(only_candidates) != 0:
        plt.scatter(only_candidates[:, 0], only_candidates[:, 1], marker=".", color='b', zorder=10,
                    label='Candidate Elements')
    if len(discarded_set) != 0:
        plt.scatter(discarded_set[:, 0], discarded_set[:, 1], marker=".", color='grey', zorder=4,
                    label='Discarded Elements')
    ax.scatter(decision[0], decision[1], marker=".", color='r', zorder=20, label='EBO')

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', '2Dplots', 'steps')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_name = 'figure0' + str(i).zfill(2) + '.png'
    file_path = os.path.join(file_path, file_name)

    plt.savefig(file_path)
    # plt.show()
    plt.close()

    im = Image.open(file_path)

    return im


def main():
    # set up data and slack vector
    num = 40
    dataset = create_random_points(2, num)
    # dataset = get_data_from_csv('inputData.csv')
    slack = [0.2, 0.2]

    ebo_tracker = EBOStreamTracker(slack)

    images = []

    # feed each element to EBO tracker

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 2))

        ebo_tracker.digest(x)

        candidate_set = ebo_tracker.candidates

        image = make_plots(dataset[:i + 1, :], candidate_set, slack, i)

        images.append(image)

    dir_path_e = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', '2Dplots')

    file_name = 'animation.gif'

    images[0].save(os.path.join(dir_path_e, file_name),
                   save_all=True,
                   append_images=images[1:],
                   duration=300,
                   loop=0)


if __name__ == "__main__":
    main()
