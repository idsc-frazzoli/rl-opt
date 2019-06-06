import os
from typing import *

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

from optimization.helpers import not_contained_in_exact, maximal_difference, no_differences
from optimization.mintracker import ExactMinTracker, ApproximateMinTracker


def create_random_points(num: int):
    """
    Generates num random datapoints between 0 and 1
    """
    data = np.random.rand(num, 3)
    return data


def comparison(dataset: np.ndarray,
               exact_min_tracker: ExactMinTracker,
               approximate_min_tracker: ApproximateMinTracker):

    for i in np.arange(len(dataset)):
        x = np.reshape(dataset[i, :], (1, 3))

        candidates_e_old = exact_min_tracker.candidates
        candiates_a_old = approximate_min_tracker.candidates

        exact_min_tracker.update_mintracker_exact(x)
        approximate_min_tracker.update_mintracker_approx(x)

    minimals, non_minimal_candidates = exact_min_tracker.get_minimals()
    minimals_a, non_minimal_candidates_a = approximate_min_tracker.get_minimals()

    comparisons_exact = exact_min_tracker.number_comparisons
    comparisons_approximate = approximate_min_tracker.number_comparisons

    max_num_candidates_e = exact_min_tracker.max_candidates
    max_num_candidates_a = approximate_min_tracker.max_candidates

    #maximal_difference(minimals, minimals_a, approximate_min_tracker.epsilon)

    return comparisons_exact, comparisons_approximate, max_num_candidates_e, max_num_candidates_a


def evaluate(m: int, sample_sizes: List[int], slack, epsilon):
    comparisons_exact_list = []
    comparisons_approx_list = []
    max_candidates_exact_list = []
    max_candidates_approx_list = []

    for size in sample_sizes:
        comparisons_exact_list_n = []
        comparisons_approx_list_n = []
        max_candidates_exact_list_n = []
        max_candidates_approx_list_n = []
        max_differences_list_n = []
        for i in range(m):
            mintracker_exact = ExactMinTracker(slack)
            mintracker_approximate = ApproximateMinTracker(slack, epsilon)
            data_set = create_random_points(size)
            comparisons_exact, comparisons_approximate, max_candidates_e, max_candidates_a = comparison(data_set, mintracker_exact,
                                                                              mintracker_approximate)
            comparisons_exact_list_n.append(comparisons_exact)
            comparisons_approx_list_n.append(comparisons_approximate)
            max_candidates_exact_list_n.append(max_candidates_e)
            max_candidates_approx_list_n.append(max_candidates_a)

        comparisons_exact_list.append(comparisons_exact_list_n)
        comparisons_approx_list.append(comparisons_approx_list_n)
        max_candidates_exact_list.append(max_candidates_exact_list_n)
        max_candidates_approx_list.append(max_candidates_approx_list_n)

        print(f'Evaluation done for sample size: {size}')

    return comparisons_exact_list, comparisons_approx_list, max_candidates_exact_list, max_candidates_approx_list


def average_best_worst(list: List[List[float]]):
    average_case = []
    best_case = []
    worst_case = []
    for times in list:
        times_as_array = np.asarray(times)
        average_case.append(np.mean(times_as_array))
        worst_case.append(np.max(times_as_array))
        best_case.append(np.min(times_as_array))

    return average_case, best_case, worst_case


def plot_num_comp(list_exact: List[List[float]], list_approx: List[List[float]], number_of_samples: List[int]):
    """
    Makes a plotly graph for the worst, best and average number of comparisons.
    :param list_exact:
    :param list_approx:
    :param number_of_samples:
    :return:
    """
    average_case_e, best_case_e, worst_case_e = average_best_worst(list_exact)

    average_case_a, best_case_a, worst_case_a = average_best_worst(list_approx)

    # Create traces
    trace0e = go.Scatter(x=number_of_samples, y=average_case_e, mode='lines+markers',
                         name='Average Case - Exact', marker=dict(size=6, color='#1067ea', ))
    trace1e = go.Scatter(x=number_of_samples, y=best_case_e, mode='lines+markers',
                         name='Best Case - Exact', marker=dict(size=6, color='#88b1ef', ))
    trace2e = go.Scatter(x=number_of_samples, y=worst_case_e, mode='lines+markers',
                         name='Worst Case - Exact', marker=dict(size=6, color='#043175', ))
    trace0a = go.Scatter(x=number_of_samples, y=average_case_a, mode='lines+markers',
                         name='Average Case - Approximate', marker=dict(size=6, color='#3da52b', ))
    trace1a = go.Scatter(x=number_of_samples, y=best_case_a, mode='lines+markers',
                         name='Best Case - Approximate', marker=dict(size=6, color='#37e519', ))
    trace2a = go.Scatter(x=number_of_samples, y=worst_case_a, mode='lines+markers',
                         name='Worst Case - Approximate', marker=dict(size=6, color='#126304', ))

    layout = go.Layout(
        title=go.layout.Title(
            text=f'Number of Comparisons during Lexicographic Semiorder Optimization',
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

    data = [trace0e, trace1e, trace2e, trace0a, trace1a, trace2a]
    fig = go.Figure(data=data, layout=layout)
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'comparison')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    filename = os.path.join(outdir, 'NumberOfComparisons.html')
    plot(fig, filename=filename)


def plot_error(list_errors: List[List[float]], number_of_samples: List[int]):
    average_case_error, best_case_error, worst_case_error = average_best_worst(list_errors)

    # Create traces
    trace0 = go.Scatter(x=number_of_samples, y=average_case_error, mode='lines+markers',
                        name='Average Case - Exact', marker=dict(size=6, color='#1067ea', ))
    trace1 = go.Scatter(x=number_of_samples, y=best_case_error, mode='lines+markers',
                        name='Best Case - Exact', marker=dict(size=6, color='#88b1ef', ))
    trace2 = go.Scatter(x=number_of_samples, y=worst_case_error, mode='lines+markers',
                        name='Worst Case - Exact', marker=dict(size=6, color='#043175', ))

    layout = go.Layout(
        title=go.layout.Title(
            text=f'Optimization',
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
                text='Maximal Error',
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
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'comparison')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    filename = os.path.join(outdir, 'MaxDifference.html')
    plot(fig, filename=filename)


def main():
    # set up data and slack vector
    slack = [0.1, 0.1, 0.1]  # slack variable to set by decision maker
    epsilon = [0.05, 0.05, 0.05]

    number_of_executions = 10000  # number of experiments
    number_of_samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000]  # number of samples

    l1, l2, l3, l4 = evaluate(number_of_executions, number_of_samples, slack, epsilon)
    plot_num_comp(l1, l2, number_of_samples)


if __name__ == "__main__":
    main()
