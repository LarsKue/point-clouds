
import torch

import matplotlib.pyplot as plt
import numpy as np


def multiscatter(samples: torch.Tensor) -> plt.Figure:
    batch_size, points, dims = samples.shape
    nrows = ncols = int(np.sqrt(batch_size))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(projection="3d"))

    for i, ax in enumerate(axes.flat):
        ax.scatter(samples[i, :, 0], samples[i, :, 1], samples[i, :, 2], s=4, alpha=0.8, color="black")

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)

    return fig
