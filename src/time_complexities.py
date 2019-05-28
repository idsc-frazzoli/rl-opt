import numpy as np
from typing import *
import time
import os
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from optimization import mintracker_purely_optimal
from optimization.helpers import not_contained_in_exact, not_contained_in_approximate, maximal_difference


def evaluate_exact(data_set: np.ndarray, slack: List[float]):
    """
    Evaluates the dataset, e.g. finds optimals of the dataset.

    :param data_set: data input
    :param slack: slack parameters used in optimization
    :return: returns minimal elements
    """
    mintracker_exact = mintracker_purely_optimal.ExactMinTracker(slack)
    for i in np.arange(len(data_set)):
        x = data_set[i, :]
        mintracker_exact.update_mintracker_exact(x)
    return mintracker_exact.get_minimals()


def evaluate_approximate(data_set: np.ndarray, slack: List[float], epsilon: List[float]):
    """
    Evaluates the dataset, e.g. finds optimals of the dataset.

    :param data_set: data input
    :param slack: slack parameters used in optimization
    :param: epsilon: epsilon approximation ratio
    :return: returns minimal elements
    """
    mintracker_approximate = mintracker_purely_optimal.ApproximateMinTracker(slack, epsilon)
    for i in np.arange(len(data_set)):
        x = data_set[i, :]
        mintracker_approximate.update_mintracker_approx(x)
    return mintracker_approximate.get_minimals()


def get_running_times(m: int, number_of_samples: List[int], slack: List[float], epsilon: List[float]):
    """
    Returns a list of running times of the lexicographic semiorder optimization.

    :param m: Number of experiments
    :param number_of_samples: number of samples used in evaluation
    :param slack: slack variables used in optimization
    :param: epsilon: epsilon approximation ratio
    :return: list of execution times
    """
    running_times_exact = []
    running_times_approximate = []
    for n in number_of_samples:
        running_times_n_exact = []
        running_times_n_approximate = []
        for i in range(m):
            data_set = np.random.rand(n, 3)

            start = time.time()
            minimals = evaluate_exact(data_set, slack)
            end = time.time()
            running_times_n_exact.append(end - start)

            start = time.time()
            minimals_a = evaluate_approximate(data_set, slack, epsilon)
            end = time.time()
            running_times_n_approximate.append(end - start)

        running_times_exact.append(running_times_n_exact)
        running_times_approximate.append(running_times_n_approximate)
        not_contained_in_approximate(minimals_a, minimals)
        different_entries = not_contained_in_exact(minimals_a, minimals)
        maximal_difference(different_entries, minimals)
        print(f'Calculations done for {n} samples')

    return running_times_exact, running_times_approximate


def make_histogram(times_list: List[List[float]], version: str):
    """
    Plots the combined histogram for teh running times for all number of samples.
    :param times_list: List of execution times
    :return:
    """
    average_case = []
    worst_case = []
    best_case = []
    for times in times_list:
        times_as_array = np.asarray(times)
        average_case.append(np.mean(times_as_array))
        worst_case.append(np.max(times_as_array))
        best_case.append(np.min(times_as_array))
        sns.distplot(times_as_array)
    # plt.show()

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'complexities', version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    filename = os.path.join(outdir, f'Histogram_{version}.png')
    plt.savefig(filename)
    plt.close()


def make_plotly(times_list: List[List[float]], number_of_samples: List[int], version: str):
    """
    Makes a plotly graph for the worst, best and average running time.

    :param times_list:
    :param number_of_samples:
    :return:
    """
    average_case = []
    worst_case = []
    best_case = []
    for times in times_list:
        times_as_array = np.asarray(times)
        average_case.append(np.mean(times_as_array))
        worst_case.append(np.max(times_as_array))
        best_case.append(np.min(times_as_array))

    # Create traces
    trace0 = go.Scatter(
        x=number_of_samples,
        y=average_case,
        mode='lines+markers',
        name='Average Case',
        marker=dict(size=6,
                    color='#2985d1',
                    )
    )
    trace1 = go.Scatter(
        x=number_of_samples,
        y=best_case,
        mode='lines+markers',
        name='Best Case',
        marker=dict(size=6,
                    color='#39ba1f',
                    )
    )
    trace2 = go.Scatter(
        x=number_of_samples,
        y=worst_case,
        mode='lines+markers',
        name='Worst Case',
        marker=dict(size=6,
                    color='#d6251b',
                    )
    )

    layout = go.Layout(
        title=go.layout.Title(
            text=f'Running Time of Lexicographic Semiorder Optimization: {version}',
            xref='paper',
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Number of Samples',
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Running Time',
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        )
    )

    data = [trace0, trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'complexities', version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    filename = os.path.join(outdir, f'RunningTimesComplexities{version}.html')
    plot(fig, filename=filename)


def main():
    slack = [0.2, 0.2, 0.2]
    epsilon = [0.05, 0.05, 0.05]
    number_of_executions = 10000  # number of experiments
    number_of_samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # number of samples
    times_exact, times_approximate = get_running_times(number_of_executions, number_of_samples, slack, epsilon)
    make_histogram(times_exact, 'Exact')
    make_plotly(times_exact, number_of_samples, 'Exact')
    make_histogram(times_approximate, 'Approximate')
    make_plotly(times_approximate, number_of_samples, 'Approximate')


if __name__ == "__main__":
    main()
