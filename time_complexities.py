import numpy as np
from typing import *
import time
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mintracker import MinTracker


def evaluate(data_set: np.ndarray, slack: List[float]):
    """
    Evaluates the dataset, e.g. finds optimals of the dataset.

    :param data_set: data input
    :param slack: slack parameters used in optimization
    :return: returns minimal elements
    """
    mintracker_lex_semi = MinTracker(slack)
    for i in np.arange(len(data_set)):
        x = np.reshape(data_set[i, :], (1, 3))
        mintracker_lex_semi.update_mintracker(x)
    return mintracker_lex_semi.get_minimals()


def get_running_times(m: int, number_of_samples: List[int], slack):
    """
    Returns a list of running times of the lexicographic semiorder optimization.

    :param m: Number of experiments
    :param number_of_samples: number of samples used in evaluation
    :param slack: slack variables used in optimization
    :return: list of execution times
    """
    running_times = []
    for n in number_of_samples:
        running_times_n = []
        for i in range(m):
            data_set = np.random.rand(n, 3)

            start = time.time()
            minimals = evaluate(data_set, slack)
            end = time.time()
            running_times_n.append(end - start)

        running_times.append(running_times_n)
        print(f'Calculations done for {n} samples')

    return running_times


def make_histogram(times_list: List[List[float]]):
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
    plt.show()
    plt.close()


def make_plotly(times_list: List[List[float]], number_of_samples: List[int]):
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
        sns.distplot(times_as_array)

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
            text='Running Time of Lexicographic Semiorder Optimization',
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
    plot(fig, filename='RunningTimesComplexities.html')


def main():
    slack = [0.1, 0.1, 0.1]
    number_of_executions = 10000                                                     # number of experiments
    number_of_samples = [100, 250, 500, 750, 1000, 1250,
                         1500, 2000, 2500, 3000, 3500,
                         4000, 4500, 5000, 5500, 6000]                              # number of samples
    times = get_running_times(number_of_executions, number_of_samples, slack)
    make_histogram(times)
    make_plotly(times, number_of_samples)


if __name__ == "__main__":
    main()