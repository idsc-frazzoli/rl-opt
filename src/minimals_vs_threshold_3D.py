import os

import matplotlib.pyplot as plt
import matplotlib2tikz
import numpy as np

from optimization import minimal_cycles, minimal_diamond, minimal_small, create_random_points


def execute(n, dim, executions, sigma):
    """

    :param n:
    :param dim:
    :param executions:
    :param sigma:
    :return:
    """
    number_minimals = []
    mins = 0.0
    minimal_dia = 0.0

    for j in range(executions):
        data_set = create_random_points(dim, n)
        minimals = minimal_cycles(data_set, sigma)

        number_minimals.append(len(minimals))
        if len(minimal_diamond(data_set, sigma)) > 0:
            minimal_dia = minimal_dia + 1.0
        if len(minimal_small(data_set, sigma)) > 0:
            mins = mins + 1.0

    return np.min(number_minimals), np.mean(number_minimals), np.max(number_minimals), mins, minimal_dia


def make_plot_num_comps(numbers_min, numbers_mean, numbers_max, slack, plot_type):
    """

    :param numbers_min:
    :param numbers_mean:
    :param numbers_max:
    :param slack:
    :return:
    """
    plt.style.use("ggplot")

    plt.xlabel("Threshold")
    plt.ylabel("Number of comparisons")

    plt.plot(slack, numbers_min, linewidth=1,
             color="green", label='Min')
    plt.plot(slack, numbers_mean, linewidth=1,
             color="blue", label='Mean')
    plt.plot(slack, numbers_max, linewidth=1,
             color="red", label='Max')

    plt.title('Number of comparisons versus threshold', pad=10)
    plt.legend(loc='lower right')

    plt.grid(True)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'minimals_vs_threshold', '3D')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    title = 'nump_comparisons'

    if plot_type == 'png':
        file_name = title + '.png'
        file_path = os.path.join(file_path, file_name)

        plt.savefig(file_path)

        # plt.show()

    if plot_type == 'tikz':
        matplotlib2tikz.save(file_path + '/' + title + ".tex")

    plt.close()


def make_plot_freq_mins(perc_min, perc_dia, slack, plot_type):
    """

    :param perc_min:
    :param perc_dia:
    :param slack:
    :return:
    """

    plt.style.use("ggplot")

    plt.xlabel("Threshold")
    plt.ylabel("Percentage ")
    plt.plot(slack, perc_min, linewidth=1,
             color="green", label='Minimals')
    plt.plot(slack, perc_dia, '--', linewidth=1,
             color="blue", label='Opt Diamond')

    plt.title('Candidate set size versus threshold', pad=10)
    plt.legend(loc='lower left')

    plt.grid(True)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'minimals_vs_threshold', '3D')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    title = 'frequency_mins'

    if plot_type == 'png':
        file_name = title + '.png'
        file_path = os.path.join(file_path, file_name)

        plt.savefig(file_path)

        # plt.show()

    if plot_type == 'tikz':
        matplotlib2tikz.save(file_path + '/' + title + ".tex")

    plt.close()


def main():
    # tikz plot as tex file or plot as png
    # plot_type = 'tikz'
    plot_type = 'png'

    dim = 3
    # sample size
    n = 10

    # set incremental increase of slack
    start = 0
    stop = 100
    increment = 10

    # number of executions for given set size and thresholds
    executions = 10

    number_min = []
    number_mean = []
    number_max = []
    percentage_min = []
    percentage_dia = []

    sigmas = []

    for i in range(start, stop + increment, increment):
        slack = i / stop
        print(slack)
        sigmas.append(slack)
        sigma = [slack, slack, slack]

        min, mean, max, mins, min_dia = execute(n, dim, executions, sigma)

        number_min.append(min / n)
        number_mean.append(mean / n)
        number_max.append(max / n)
        percentage_min.append(mins / n)
        percentage_dia.append(min_dia / n)

    make_plot_num_comps(number_min, number_mean, number_max, sigmas, plot_type)
    make_plot_freq_mins(percentage_min, percentage_dia, sigmas, plot_type)


if __name__ == "__main__":
    main()
