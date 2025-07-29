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
        for idx, y_i in enumerate(y):
            e = np.abs(y_i.T - steady_state[:, idx : idx + 1])
            e_max = np.max(e, axis=1).reshape(-1, 1)
            transient_time = np.argmax(e < e_max * SETTLING_TIME_THRESHOLD, axis=1)
            if plot:
                for idx, _ in enumerate(e_max):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(t, y_i[:, idx], label=f"Output {idx + 1}")
                    ax.plot(t, e[idx, :], label="|y - y_ss|")
                ax.plot(
                    t,
                    e_max[idx] * SETTLING_TIME_THRESHOLD * np.ones_like(t),
                    label="e_max * 0.02",
                    linestyle="--",
                )
                ax.scatter(
                    transient_time[idx] * ss.dt,
                    e[idx, transient_time[idx]],
                    color="red",
                    label=f"Transient Time {transient_time[idx]}",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Output")
                ax.legend()
                ax.grid()
                plt.savefig(f"step_{n}.png")

        return np.argmax(e < e_max * SETTLING_TIME_THRESHOLD, axis=1), e, e_max

    if not ss.is_stable():
        raise ValueError("The system is not stable, transient time cannot be computed.")

    steady_state = get_steady_state(ss)
    if n_max < 100:
        raise ValueError("n_max should be at least 100 for transient time calculation.")
    n = 100
    transient_time, e, e_max = get_transient_from_step_response(n, plot=plot)
    while np.all(e > e_max * SETTLING_TIME_THRESHOLD) and n < n_max:
        n += 100
        transient_time, e, e_max = get_transient_from_step_response(n, plot=plot)

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
