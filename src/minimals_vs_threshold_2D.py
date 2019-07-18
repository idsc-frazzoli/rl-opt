import os

import matplotlib.pyplot as plt
import matplotlib2tikz
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from optimization import minimal_cycles
from optimization.helpers import set_subtraction, create_random_points


def execute(n, dim, executions, sigma):
    number_minimals = []
    for j in range(executions):
        data_set = create_random_points(dim, n)
        minimals = minimal_cycles(data_set, sigma)

        number_minimals.append(len(minimals))
        if (len(minimals) / n > 0.8) and ((sigma[0] < 0.4) or (sigma[0] < 0.4)):
            plot_cycles(data_set, minimals, sigma)

    return np.min(number_minimals), np.mean(number_minimals), np.max(number_minimals)


def plot_cycles(data_set, minimals, slack):
    """
    Create plots for each new point
    """
    # set up plot figure
    fig = plt.figure()
    ax = fig.add_axes([0.18, 0.23, 0.64, 0.64])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Setting the axes properties

    ax.set_xlabel(f'Score of objective 1 ($\sigma = {slack[0]}$)')
    ax.set_ylabel(f'Score of objective 2 ($\sigma = {slack[1]}$)')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Optimal elements',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='x', color='w', label='Non-optimal elements',
                              markerfacecolor='grey', markersize=10)]

    plt.title('Bottom Cycles', pad=10)

    ax.legend(handles=legend_elements, bbox_to_anchor=(0., -0.3, 1., .102), loc=10,
              ncol=2, borderaxespad=0.)

    non_minimal = set_subtraction(data_set, minimals)

    if non_minimal.size != 0:
        plt.scatter(non_minimal[:, 0], non_minimal[:, 1], marker="x", color='grey', zorder=10,
                    label='Non-Minimal Elements')

    ax.scatter(minimals[:, 0], minimals[:, 1], marker=".", color='r', zorder=20, label='Minimal Elements')

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'minimals_vs_threshold', '2D',
                             'cycles')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    file_name = 'figure' + 'sigma1_' + str(slack[0]) + 'sigma2_' + str(slack[1]) + '.png'
    file_path = os.path.join(file_path, file_name)

    plt.savefig(file_path)
    # plt.show()
    plt.close()


def plot_heatmap(data_frame: pd.DataFrame, slacks, title: str, plot_type):
    slacks_data = pd.DataFrame({'Slack': slacks})
    data_frame['Slack'] = slacks_data
    data_frame = data_frame.set_index('Slack')

    data_frame.columns = slacks

    data_frame = data_frame.transpose()
    data_frame = data_frame[::-1]

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(data_frame, linewidths=.5, ax=ax, cmap='viridis')
    plt.title(title)
    ax.set_xlabel("Threshold of first objective")
    ax.set_ylabel("Threshold of second objective")

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'minimals_vs_threshold', '2D')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    if plot_type == 'png':
        file_name = title + '.png'
        file_path = os.path.join(file_path, file_name)

        plt.savefig(file_path)

        plt.show()

    if plot_type == 'tikz':
        matplotlib2tikz.save(file_path + '/' + title + ".tex")


def generate_data(stop, increment):
    dim = 2
    n = 10

    executions = 10

    overall_min = []
    overall_mean = []
    overall_max = []
    slack_string = []

    for i in range(0, stop + increment, increment):
        slack1 = i / stop
        slack_string.append(slack1)
        number_min = []
        number_mean = []
        number_max = []
        for q in range(0, stop + increment, increment):
            slack2 = q / stop
            sigma = [slack1, slack2]

            min, mean, max = execute(n, dim, executions, sigma)
            print(sigma)

            number_min.append(min / n)
            number_mean.append(mean / n)
            number_max.append(max / n)

        overall_min.append(number_min)
        overall_mean.append(number_mean)
        overall_max.append(number_max)

    data_min = pd.DataFrame(overall_min)
    # data_min.to_pickle('min.pkl')

    data_mean = pd.DataFrame(overall_mean)
    # data_mean.to_pickle('mean.pkl')

    data_max = pd.DataFrame(overall_max)
    # data_max.to_pickle('max.pkl')

    return data_min, data_mean, data_max


def reuse_data():
    data_min = pd.read_pickle('min.pkl')
    data_mean = pd.read_pickle('mean.pkl')
    data_max = pd.read_pickle('max.pkl')

    return data_min, data_mean, data_max


def main():
    start = 0
    stop = 100
    increment = 20

    slack_string = []

    for i in range(start, stop + increment, increment):
        slack1 = i / stop
        slack_string.append(slack1)

    data_min, data_mean, data_max = generate_data(stop, increment)
    # data_min, data_mean, data_max = reuse_data()

    plot_type = 'tikz'
    plot_type = 'png'

    plot_heatmap(data_min, slack_string, 'Best case', plot_type)
    plot_heatmap(data_mean, slack_string, 'Average case', plot_type)
    plot_heatmap(data_max, slack_string, 'Worst case', plot_type)


if __name__ == "__main__":
    main()
