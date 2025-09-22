"""
Base analysis functionality to avoid circular imports between models.base and systemtheory.analysis.

This module contains analysis functions and classes that need to be used by both
the models and systemtheory modules, following the repository's naming convention
of using 'base.py' for foundational components.
"""

import dataclasses
from typing import Optional, Tuple, Union

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
import torch

from ..utils import base as utils
from ..utils import transformation as trans


@dataclasses.dataclass
class SectorBounded:
    """Sector bounded nonlinearity parameters."""

    alpha: float
    beta: float


@dataclasses.dataclass
class AdditionalParameters:
    """Additional parameters returned by L2 analysis."""

    Lambda: NDArray[np.float64]
    X: NDArray[np.float64]
    ga2: np.float64


@dataclasses.dataclass
class LinearNp:
    """NumPy-based linear system representation."""

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    C: NDArray[np.float64]
    D: NDArray[np.float64]


@dataclasses.dataclass
class UncertainLti(LinearNp):
    """Uncertain linear time-invariant system representation."""

    B2: NDArray[np.float64]
    D12: NDArray[np.float64]
    C2: NDArray[np.float64]
    D21: NDArray[np.float64]


class AnalysisLti:
    """L2 stability analysis for linear time-invariant systems."""

    def __init__(
        self,
        linear_system: Union[torch.nn.Module, LinearNp],
        nz: Optional[int] = None,
        sdp_opt: str = cp.MOSEK,
    ):
        """
        Initialize the LTI analysis.

        Args:
            linear_system: Either a torch.nn.Module with A,B,C,D attributes or LinearNp dataclass
            nz: Size of nonlinear channel (defaults to nx)
            sdp_opt: SDP optimization solver (default: MOSEK)
        """
        self.sdp_opt = sdp_opt

        # Extract system matrices - handle both torch and numpy systems
        if hasattr(linear_system, "A") and hasattr(linear_system.A, "cpu"):
            # Torch-based system (from models.base.Linear)
            A = linear_system.A.cpu().detach().numpy()
            B = linear_system.B.cpu().detach().numpy()
            C = linear_system.C.cpu().detach().numpy()
            D = linear_system.D.cpu().detach().numpy()
        elif isinstance(linear_system, LinearNp):
            # NumPy-based system
            A, B, C, D = (
                linear_system.A,
                linear_system.B,
                linear_system.C,
                linear_system.D,
            )
        else:
            # Assume it has numpy arrays as attributes
            A, B, C, D = (
                linear_system.A,
                linear_system.B,
                linear_system.C,
                linear_system.D,
            )

        self.nx, self.nd, self.ne = A.shape[0], B.shape[1], C.shape[0]
        if not nz:
            self.nz = self.nx
        else:
            self.nz = nz
        self.nw = self.nz

        self.theta = UncertainLti(
            A,
            B,
            C,
            D,
            np.zeros((self.nx, self.nw)),
            np.zeros((self.ne, self.nw)),
            np.zeros((self.nw, self.nx)),
            np.zeros((self.nw, self.nd)),
        )

    def l2(
        self, uncertainty: SectorBounded, ga2: Optional[np.float64] = None
    ) -> Tuple[str, AdditionalParameters]:
        """
        Perform L2 stability analysis.

        Args:
            uncertainty: Sector bounded nonlinearity parameters
            ga2: L2 gain bound (if None, will be optimized)

        Returns:
            Tuple of (status_message, additional_parameters)
        """
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
            print(f"cvxpy failed. {cp.SolverError}")
            return (
                problem.status,
                AdditionalParameters(
                    np.eye(self.nw), np.eye(self.nx), np.array([[1.0]])
                ),
            )
        if not problem.status == "optimal":
            print(f"cvxpy did not find a solution. {problem.status}")
            return (
                problem.status,
                AdditionalParameters(
                    np.eye(self.nw), np.eye(self.nx), np.array([[1.0]])
                ),
            )
        print(
            f"norm L {np.linalg.norm(L.value, 2)}, norm X {np.linalg.norm(X.value, 2)}"
        )

        # assert self.sanity_check(X.value, ga2.value, L.value, uncertainty)
        assert self._sanity_check(
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

    def _sanity_check(self, X, ga2, L, uncertainty) -> bool:
        """Sanity check for the L2 analysis solution."""
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
