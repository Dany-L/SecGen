import dataclasses
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
import scipy.signal as sig
import matplotlib.pyplot as plt

from ..models import base
from ..utils import base as utils
from ..utils import transformation as trans

SETTLING_TIME_THRESHOLD = 0.1


@dataclasses.dataclass
class LinearNp:
    A: NDArray[np.float64]
    B: NDArray[np.float64]
    C: NDArray[np.float64]
    D: NDArray[np.float64]


@dataclasses.dataclass
class SectorBounded:
    alpha: float
    beta: float


@dataclasses.dataclass
class UncertainLti(LinearNp):
    B2: NDArray[np.float64]
    D12: NDArray[np.float64]
    C2: NDArray[np.float64]
    D21: NDArray[np.float64]


@dataclasses.dataclass
class AdditionalParameters:
    Lambda: NDArray[np.float64]
    X: NDArray[np.float64]
    ga2: np.float64


class AnalysisLti:
    def __init__(
        self, ss: base.Linear, nz: Optional[int] = None, sdp_opt: str = cp.MOSEK
    ):
        self.sdp_opt = sdp_opt
        self.nx, self.nd, self.ne = ss.A.shape[0], ss.B.shape[1], ss.C.shape[0]
        if not nz:
            self.nz = self.nx
        else:
            self.nz = nz
        self.nw = self.nz

        self.theta = UncertainLti(
            ss.A.cpu().detach().numpy(),
            ss.B.cpu().detach().numpy(),
            ss.C.cpu().detach().numpy(),
            ss.D.cpu().detach().numpy(),
            np.zeros((self.nx, self.nw)),
            np.zeros((self.ne, self.nw)),
            np.zeros((self.nw, self.nx)),
            np.zeros((self.nw, self.nd)),
        )

    def l2(
        self, uncertainty: SectorBounded, ga2: Optional[np.float64] = None
    ) -> Tuple[str, AdditionalParameters]:
        X = cp.Variable((self.nx, self.nx), symmetric=True)
        if not ga2:
            ga2 = cp.Variable((1, 1))
        if self.nd == 1:
            ga2_m = -ga2
        else:
            ga2_m = -ga2 * np.eye(self.nd)

        # la = cp.Variable((self.nw, 1))
        la = np.ones((self.nw, 1))
        L = cp.diag(la)
        L1 = trans.bmat(
            [
                [
                    np.eye(self.nx),
                    np.zeros((self.nx, self.nd)),
                    np.zeros((self.nx, self.nw)),
                ],
                [self.theta.A, self.theta.B, self.theta.B2],
            ]
        )
        L2 = trans.bmat(
            [
                [
                    np.zeros((self.nd, self.nx)),
                    np.eye(self.nd),
                    np.zeros((self.nd, self.nw)),
                ],
                [self.theta.C, self.theta.D, self.theta.D12],
            ]
        )
        L3 = trans.bmat(
            [
                [
                    np.zeros((self.nw, self.nx)),
                    np.zeros((self.nw, self.nd)),
                    np.eye(self.nw),
                ],
                [self.theta.C2, self.theta.D21, np.zeros((self.nw, self.nz))],
            ]
        )
        P_r = trans.bmat(
            [
                [-np.eye(self.nw), uncertainty.beta * np.eye(self.nw)],
                [np.eye(self.nw), -uncertainty.alpha * np.eye(self.nw)],
            ]
        )
        P = (
            P_r.T
            @ cp.bmat(
                [[np.zeros((self.nw, self.nw)), L.T], [L, np.zeros((self.nw, self.nw))]]
            )
            @ P_r
        )

        F = (
            L1.T
            @ cp.bmat(
                [[-X, np.zeros((self.nx, self.nx))], [np.zeros((self.nx, self.nx)), X]]
            )
            @ L1
            + L2.T
            @ cp.bmat(
                [
                    [ga2_m, np.zeros((self.nd, self.ne))],
                    [np.zeros((self.ne, self.nd)), np.eye(self.ne)],
                ]
            )
            @ L2
            + L3.T @ P @ L3
        )
        nF = F.shape[0]
        eps = 1e-3
        problem = cp.Problem(
            cp.Minimize(cp.norm(X)),
            [
                F << -eps * np.eye(nF),
                X >> eps * np.eye(self.nx),
            ],
        )

        try:
            problem.solve(solver=self.sdp_opt)
        except cp.SolverError:
            Warning(f"cvxpy failed. {cp.SolverError}")
            return (
                problem.status,
                AdditionalParameters(
                    np.eye(self.nw), np.eye(self.nx), np.array([[1.0]])
                ),
            )
        if not problem.status == "optimal":

            Warning(f"cvxpy did not find a solution. {problem.status}")
            return (
                problem.status,
                AdditionalParameters(
                    np.eye(self.nw), np.eye(self.nx), np.array([[1.0]])
                ),
            )
        print(f"norm L {np.linalg.norm(L.value,2)}, norm X {np.linalg.norm(X.value,2)}")

        # assert self.sanity_check(X.value, ga2.value, L.value, uncertainty)
        assert self.sanity_check(
            X.value, utils.get_opt_values(ga2), L.value, uncertainty
        )

        return (
            problem.status,
            AdditionalParameters(
                utils.get_opt_values(L),
                utils.get_opt_values(X),
                utils.get_opt_values(ga2),
            ),
        )

    def sanity_check(self, X, ga2, L, uncertainty) -> bool:
        L1 = trans.bmat(
            [
                [
                    np.eye(self.nx),
                    np.zeros((self.nx, self.nd)),
                    np.zeros((self.nx, self.nw)),
                ],
                [self.theta.A, self.theta.B, self.theta.B2],
            ]
        )
        L2 = trans.bmat(
            [
                [
                    np.zeros((self.nd, self.nx)),
                    np.eye(self.nd),
                    np.zeros((self.nd, self.nw)),
                ],
                [self.theta.C, self.theta.D, self.theta.D12],
            ]
        )
        L3 = trans.bmat(
            [
                [
                    np.zeros((self.nw, self.nx)),
                    np.zeros((self.nw, self.nd)),
                    np.eye(self.nw),
                ],
                [self.theta.C2, self.theta.D21, np.zeros((self.nw, self.nz))],
            ]
        )
        P_r = trans.bmat(
            [
                [-np.eye(self.nw), uncertainty.beta * np.eye(self.nw)],
                [np.eye(self.nw), -uncertainty.alpha * np.eye(self.nw)],
            ]
        )
        P = (
            P_r.T
            @ trans.bmat(
                [[np.zeros((self.nw, self.nw)), L.T], [L, np.zeros((self.nw, self.nw))]]
            )
            @ P_r
        )

        F = (
            L1.T
            @ trans.bmat(
                [[-X, np.zeros((self.nx, self.nx))], [np.zeros((self.nx, self.nx)), X]]
            )
            @ L1
            + L2.T
            @ trans.bmat(
                [
                    [-ga2, np.zeros((self.nd, self.ne))],
                    [np.zeros((self.ne, self.nd)), np.eye(self.ne)],
                ]
            )
            @ L2
            + L3.T @ P @ L3
        )

        try:
            np.linalg.cholesky(-F)
        except np.linalg.LinAlgError:
            return False

        A_tilde = X @ self.theta.A
        B_tilde = X @ self.theta.B
        B2_tilde = X @ self.theta.B2

        C2_tilde = L @ self.theta.C2
        D21_tilde = L @ self.theta.D21

        M11 = trans.bmat(
            [
                [-X, np.zeros((self.nx, self.nd)), C2_tilde.T],
                [np.zeros((self.nd, self.nx)), -ga2 * np.eye(self.nd), D21_tilde.T],
                [C2_tilde, D21_tilde, -2 * L],
            ]
        )
        M21 = trans.bmat(
            [[A_tilde, B_tilde, B2_tilde], [self.theta.C, self.theta.D, self.theta.D12]]
        )
        M22 = trans.bmat(
            [
                [-X, np.zeros((self.nx, self.ne))],
                [np.zeros((self.ne, self.nx)), -np.eye(self.ne)],
            ]
        )
        M = trans.bmat([[M11, M21.T], [M21, M22]])
        try:
            np.linalg.cholesky(-M)
        except np.linalg.LinAlgError:
            return False
        return True


def get_transient_time(ss: base.Linear) -> np.float64:

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
        e = np.abs(y[0].T - steady_state)
        e_max = np.max(e, axis=1).reshape(-1, 1)
        transient_time = np.argmax(e < e_max * SETTLING_TIME_THRESHOLD, axis=1)
        if plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, y[0])
            for e_i in e_max:
                ax.plot(t, np.ones_like(t) * e_i - e_i * SETTLING_TIME_THRESHOLD)
            for i, e_i in enumerate(transient_time):
                ax.scatter(e_i * ss.dt, y[0][e_i, i])
            plt.savefig(f"step_{n}.png")

        return np.argmax(e < e_max * SETTLING_TIME_THRESHOLD, axis=1), e, e_max

    steady_state = get_steady_state(ss)
    n = 100
    transient_time, e, e_max = get_transient_from_step_response(n)
    while np.all(e > e_max * SETTLING_TIME_THRESHOLD) or n > 1000:
        n += 100
        transient_time, e, e_max = get_transient_from_step_response(n, plot=False)

    if n > 1000:
        raise ValueError("No reasonable transient time could be calculated.")

    return transient_time * ss.dt


def get_steady_state(
    ss: base.Linear, u: NDArray[np.float64] = None
) -> NDArray[np.float64]:
    nd = ss.B.shape[1]
    if u is None:
        u = np.ones((nd, 1))
    nx = ss.A.shape[0]
    return (
        -ss.C.cpu().detach().numpy()
        @ np.linalg.inv(ss.A.cpu().detach().numpy() - np.eye(nx))
        @ ss.B.detach().numpy()
        + ss.D.cpu().detach().numpy()
    ) @ u
