import os

import matplotlib.pyplot as plt
import matplotlib2tikz
import numpy as np

from optimization.ebo_stream import EBOStreamTracker
from optimization.helpers import create_random_points


def execute(n, dim, executions, sigma):
    number_max_candidates = []
    number_comps = []
    for j in range(executions):
        mintracker_exact = EBOStreamTracker(sigma)
        data_set = create_random_points(dim, n)
        for i in np.arange(len(data_set)):
            x = np.reshape(data_set[i, :], (1, dim))

            mintracker_exact.digest(x)

        cand_set_size = mintracker_exact.max_candidates
        number_max_candidates.append(cand_set_size)
        number_comps.append(mintracker_exact.number_comparisons)

    return np.min(number_max_candidates), np.mean(number_max_candidates), np.max(number_max_candidates), \
           np.min(number_comps), np.mean(number_comps), np.max(number_comps)


def make_plots(numbers_min, numbers_mean, numbers_max, samples, title, plot_type):
    """
    Create plots for each new point
    """
    plt.style.use("ggplot")
    # set up plot figure

    plt.xlabel("Sample size")
    if title == 'candidate_set_size':
        plt.ylabel("Percentage")
    else:
        plt.ylabel("Number of comparisons")

    plt.plot(samples, numbers_min, linewidth=1,
             color="green", label='Min')
    plt.plot(samples, numbers_mean, linewidth=1,
             color="blue", label='Mean')
    plt.plot(samples, numbers_max, linewidth=1,
             color="red", label='Max')

    plt.title('Candidate set size versus threshold', pad=10)
    plt.legend(loc='lower right')

    plt.grid(True)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'candidate_set_size')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    if plot_type == 'png':
        file_name = title + '.png'
        file_path = os.path.join(file_path, file_name)

        plt.savefig(file_path)

        # plt.show()

    if plot_type == 'tikz':
        matplotlib2tikz.save(file_path + '/' + title + ".tex")

    plt.close()


def main():
    dim = 3

    executions = 1000

    samples = [10, 20, 30, 40, 50]
    # samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
    #            9000, 9500, 10000]

    # tikz plot as tex file or plot as png
    plot_type = 'tikz'
    # plot_type = 'png'

    number_min = []
    number_mean = []
    number_max = []

    comps_min = []
    comps_mean = []
    comps_max = []

    for i in samples:
        n = i
        slack = 0.2
        sigma = [slack, slack, slack]

        min_size, mean_size, max_size, min_comps, mean_comps, max_comps = execute(n, dim, executions, sigma)

        number_min.append(min_size)
        number_mean.append(mean_size)
        number_max.append(max_size)

        comps_min.append(min_comps)
        comps_mean.append(mean_comps)
        comps_max.append(max_comps)

        # print(i)

    make_plots(number_min, number_mean, number_max, samples, 'candidate_set_size', plot_type)
    make_plots(comps_min, comps_mean, comps_max, samples, "number_comparisons", plot_type)


if __name__ == "__main__":
    main()
