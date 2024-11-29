import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

# import tikzplotlib
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_sequence(
    sequence: List[NDArray],
    dt: float,
    title: Optional[str] = None,
    legend: Optional[List[str]] = None,
) -> Figure:
    N, n = sequence[0].shape
    t = np.linspace(0, (N - 1) * dt, N)
    if legend is None:
        legend = ["unknown" for n in range(n)]
    fig, axs = plt.subplots(nrows=1, ncols=n, tight_layout=True, squeeze=False)
    for state_idx, (ax, label) in enumerate(zip(axs.reshape(-1), legend)):
        for seq in sequence:
            ax.plot(t, seq[:, state_idx], label=label)
        ax.set_xlabel(f"time, dt = {dt:.2f}")
        if title is not None:
            ax.set_title(f"{title} state {state_idx}")
        ax.grid()
        # ax.legend()
    return fig


def save_fig(fig: Figure, name: str, directory: str) -> None:
    # save png
    fig.savefig(os.path.join(directory, f"{name}.jpg"))
    # save tex
    # tikzplotlib.save(os.path.join(directory, f"{name}.tex"))
    plt.close(fig)
