import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

from optimization import minimal_cycles, set_subtraction, ebo, get_data_from_csv


def make_plots(data_set, minimals, slack, step):
    """
    Create plots for each new point
    """
    # set up plot figure and axes
    fig = plt.figure()
    ax = fig.add_axes([0.18, 0.23, 0.64, 0.64])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(f'Objective 1 ($\sigma = {slack[0]}$)')
    ax.set_ylabel(f'Objective 2 ($\sigma = {slack[1]}$)')

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Minimal Elements',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Non-Minimal Elements',
                              markerfacecolor='grey', markersize=10)]

    plt.title('Bottom Cycles and EBO', pad=10)

    ax.legend(handles=legend_elements, bbox_to_anchor=(0., -0.3, 1., .102), loc=10,
              ncol=3, borderaxespad=0.)

    non_minimal = set_subtraction(data_set, minimals)

    if non_minimal.size != 0:
        plt.scatter(non_minimal[:, 0], non_minimal[:, 1], marker=".", color='grey', zorder=10,
                    label='Non-Minimal Elements')

    ax.plot(np.append(minimals[:, 0], minimals[0, 0]), np.append(minimals[:, 1], minimals[0, 1]), linewidth=1,
            color="lightcoral")
    ax.plot([minimals[-1, 0], minimals[0, 0]], [minimals[-1, 1], minimals[0, 1]], linewidth=1,
            color="lightcoral")

    # create vectors for threshold length for each minimal point
    for i in range(len(minimals)):
        x_value = minimals[i, 0]
        y_value = minimals[i, 1]
        ax.arrow(x_value, y_value, slack[0], 0, color="powderblue", length_includes_head=True,
                 head_width=0.01, head_length=0.02)
        ax.arrow(x_value, y_value, 0, slack[1], color="powderblue", length_includes_head=True,
                 head_width=0.01, head_length=0.02)

    ax.scatter(minimals[:, 0], minimals[:, 1], marker=".", color='r', zorder=20, label='Minimal Elements')

    # ebo decision
    decision = ebo(data_set, slack)

    circle = plt.Circle((decision[0], decision[1]), 0.025, color='forestgreen', fill=False)

    plt.gca().add_artist(circle)

    # save file
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'minimals_vs_ebo')
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_name = 'figure' + str(step).zfill(3) + '.png'
    file_path = os.path.join(file_path, file_name)

    plt.savefig(file_path)
    # plt.show()
    plt.close()

    im = Image.open(file_path)

    return im


def main():
    n = 50
    # data_values = create_random_points(dim, n)
    data_values = get_data_from_csv('inputData.csv')
    sigma = [0.2, 0.2]

    images = []

    for i in range(1, n + 1):
        data_set = data_values[:i, :]
        minimals = minimal_cycles(data_set, sigma)

        im = make_plots(data_set, minimals, sigma, i)
        images.append(im)

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 'minimals_vs_ebo')

    file_name = 'animation.gif'

    images[0].save(os.path.join(dir_path, file_name),
                   save_all=True,
                   append_images=images[1:],
                   duration=400,
                   loop=0)


if __name__ == "__main__":
    main()
