from typing import Optional

import numpy as np
from numpy.typing import NDArray
import scipy.signal as sig
import matplotlib.pyplot as plt

from ..models import base as models_base

SETTLING_TIME_THRESHOLD = 0.02


def get_transient_time(ss: models_base.Linear, n_max=5000, plot=False) -> np.float64:

    def get_transient_from_step_response(n, plot=False):
        t, y = sig.dstep(
            (
                ss.A.detach().numpy(),
                ss.B.detach().numpy(),
                ss.C.detach().numpy(),
                ss.D.detach().numpy(),
                ss.dt,
            ),
            n=n,
        )
        transient_times = []
        for n_u in range(len(y)):
            # iterate over inputs
            ss_i = steady_state[:, n_u : n_u + 1]
            for n_y in range(y[0].shape[1]):
                # iterate over outputs
                y_i = y[n_u][:, n_y]
                e_i = np.abs(y_i - ss_i[n_y])
                e_max = np.max(e_i)
                band = SETTLING_TIME_THRESHOLD * e_max
                lower, upper = ss_i[n_y] - band, ss_i[n_y] + band
                inside = (y_i >= lower) & (y_i <= upper)  # boolean mask
                for k in range(len(y_i)):
                    if inside[k] and inside[k:].all():  # enters and stays
                        transient_time = k
                    else:
                        transient_time = -1
                transient_times.append(transient_time)

                if plot:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(t, y_i, label=f"Output {n_y + 1}")
                    ax.plot(t, e_i, label="|y - y_ss|")
                    ax.plot(
                        t,
                        e_max * SETTLING_TIME_THRESHOLD * np.ones_like(t),
                        label="e_max * 0.02",
                        linestyle="--",
                    )
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Output")
                    ax.legend()
                    ax.grid()
                    plt.savefig(f"step_{n}_output_{n_y + 1}.png")

        return np.max(transient_times)

    if not ss.is_stable():
        raise ValueError("The system is not stable, transient time cannot be computed.")

    steady_state = get_steady_state(ss)
    if n_max < 100:
        raise ValueError("n_max should be at least 100 for transient time calculation.")
    n = 100
    transient_time = get_transient_from_step_response(n, plot=plot)
    while transient_time < 0 and n < n_max:
        n += 100
        transient_time = get_transient_from_step_response(n, plot=plot)

    if n >= n_max:
        transient_time = 100  # default value if transient time cannot be determined

    return transient_time * ss.dt.cpu().detach().numpy()


def get_steady_state(
    ss: models_base.Linear, u: Optional[NDArray[np.float64]] = None
) -> NDArray[np.float64]:
    nd = ss.B.shape[1]
    ne = ss.C.shape[0]
    nx = ss.A.shape[0]
    steady_state = np.zeros((ne, nd))
    for nd_i in range(nd):
        u = np.zeros((nd, 1))
        u[nd_i] = 1.0  # unit step for each input channel
        steady_state[:, nd_i : nd_i + 1] = (
            -ss.C.cpu().detach().numpy()
            @ np.linalg.inv(ss.A.cpu().detach().numpy() - np.eye(nx))
            @ ss.B.detach().numpy()
            + ss.D.cpu().detach().numpy()
        ) @ u
    return steady_state
