import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from optimization.mintracker import ExactMinTracker, ApproximateMinTracker


# import data from csv as np.array
def get_data_from_csv():
    """
    Retrieve Data from csv file
    """
    file_name = os.path.join('resources', 'inputData.csv')
    df = pd.read_csv(file_name)
    return df.values


def create_random_points(num):
    """
    Generates num random datapoints between 0 and 1
    """
    datavalues = np.random.rand(num, 2)
    return datavalues


def get_lower_bound(non_discarded_set):
    """
    Get lower bound of retainer set for creation of rectangle in plot
    """
    lex_sorted_set = non_discarded_set[np.lexsort(np.rot90(non_discarded_set))]
    current_y_min = lex_sorted_set[0, 1]
    y_sort = [0]
    for i in range(1, len(lex_sorted_set)):
        if lex_sorted_set[i, 1] < current_y_min:
            y_sort.append(i)
            current_y_min = lex_sorted_set[i, 1]
    return lex_sorted_set[y_sort, :]


def make_plots(discarded_set, non_minimal_candidate_set, minimal_set, lower_bound, slack, i, version: str):
    """
    Create plots for each new point
    """
    # set up plot figure
    fig = plt.figure()
    ax = fig.add_axes([0.18, 0.23, 0.64, 0.64])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Setting the axes properties

    ax.set_xlabel(f'Objective 1 ($\sigma = {slack[0]}$)')
    ax.set_ylabel(f'Objective 2 ($\sigma = {slack[0]}$)')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Minimal Elements',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Candidate Elements',
                              markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Discarded Elements',
                              markerfacecolor='y', markersize=10)]

    plt.title('Lexicographic Survivor Set', pad=10)

    ax.legend(handles=legend_elements, bbox_to_anchor=(0., -0.3, 1., .102), loc=10,
           ncol=3, borderaxespad=0.)

    if non_minimal_candidate_set.size != 0:
        plt.scatter(non_minimal_candidate_set[:, 0], non_minimal_candidate_set[:, 1], marker=".", color='g', zorder=10, label='Candidate Elements')
    if discarded_set is not None:
        plt.scatter(discarded_set[:, 0], discarded_set[:, 1], marker=".", color='y', zorder=4, label='Discarded Elements')
    ax.scatter(minimal_set[:, 0], minimal_set[:, 1], marker=".", color='b', zorder=20, label='Minimal Elements')

    for j in range(len(lower_bound) - 1):
        diff = lower_bound[j + 1, 0] - lower_bound[j, 0]
        rectangle = plt.Rectangle((lower_bound[j, 0], lower_bound[j, 1]), diff, slack[1], alpha=0.5,
                                  fc="CornflowerBlue", zorder=1)
        plt.gca().add_patch(rectangle)

    diff = lower_bound[0, 0] + slack[0] - lower_bound[-1, 0]
    rectangle = plt.Rectangle((lower_bound[-1, 0], lower_bound[-1, 1]), diff, slack[1], alpha=0.5, fc="CornflowerBlue",
                              zorder=1)
    plt.gca().add_patch(rectangle)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', '2Dplots', version)
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_name = 'figure0' + str(i).zfill(2) + '.png'
    file_path = os.path.join(file_path, file_name)

    plt.savefig(file_path)
    #plt.show()
    plt.close()


def main():
    # set up data and slack vector
    num = 100
    #dataset = create_random_points(num)
    dataset = get_data_from_csv()
    slack = [0.2, 0.2]
    epsilon = [0.05, 0.05]

    mintracker_exact = ExactMinTracker(slack)
    mintracker_approximate = ApproximateMinTracker(slack, epsilon)

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 2))

        mintracker_exact.update_mintracker(x)
        mintracker_approximate.update_mintracker(x)

        minimals, non_minimal_candidates = mintracker_exact.get_minimals()
        minimals_a, non_minimal_candidates_a = mintracker_approximate.get_minimals()

        lower_bound = get_lower_bound(mintracker_exact.candidates)
        lower_bound_a = get_lower_bound(mintracker_exact.candidates)

        make_plots(mintracker_exact.discarded, non_minimal_candidates, minimals,
                   lower_bound, slack, i, 'exact')
        make_plots(mintracker_approximate.discarded, non_minimal_candidates_a, minimals_a,
                   lower_bound_a, slack, i, 'approximate')


if __name__ == "__main__":
    main()