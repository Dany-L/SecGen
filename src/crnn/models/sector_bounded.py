from typing import Callable, List, Literal

import cvxpy as cp
import numpy as np
import torch

from .. import tracker as base_tracker
from ..utils import transformation as trans
from . import base


class SectorBoundedLtiRnn(base.ConstrainedModule):
    def __init__(
        self,
        nd: int,
        ne: int,
        nz: int,
        optimizer: str = cp.MOSEK,
        nonlinearity: Literal["tanh", "relu", "deadzone", "sat"] = "tanh",
        tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
    ) -> None:
        super().__init__(nz, nd, ne, optimizer, nonlinearity, tracker)
        self.tracker = tracker

    def sdp_constraints(self) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            M11 = trans.torch_bmat(
                [[-self.X, self.C2.T], [self.C2, -2 * torch.eye(self.nz)]]
            )
            M21 = trans.torch_bmat([[self.A_tilde, self.B2_tilde]])
            M22 = -self.X
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        return [stability_lmi]

    def initialize_parameters(self) -> None:

        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, C2.T], [C2, -2 * np.eye(self.nz)]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        problem = cp.Problem(cp.Minimize([None]), [M << -eps * np.eye(nM)])
        problem.solve(solver=self.optimizer)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")
        self.tracker.track(
            base_tracker.Log(
                "",
                f"Feasible initial parameter set found, write back parameters, problem status {problem.status}",
            )
        )

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)

    def project_parameters(self) -> None:
        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, C2.T], [C2, -2 * np.eye(self.nz)]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        M0 = -self.sdp_constraints()[0]().detach().numpy()
        problem = cp.Problem(cp.Minimize(cp.norm(M0 - M)), [M << -eps * np.eye(nM)])
        problem.solve(solver=self.optimizer)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")
        self.tracker.track(
            base_tracker.Log(
                "",
                f"Project parameters to feasible set d: {np.linalg.norm(M0-M.value):.2f}, write back parameters, problem status {problem.status}",
            )
        )

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)

    def check_constraints(self) -> bool:
        # check if constraints are psd
        with torch.no_grad():
            for lmi in self.sdp_constraints():
                _, info = torch.linalg.cholesky_ex(lmi())
                if info > 0:
                    return False
        return True


class GeneralSectorBoundedLtiRnn(base.ConstrainedModule):
    def __init__(
        self,
        nz: int,
        nd: int,
        ne: int,
        optimizer: str = cp.MOSEK,
        nonlinearity: Literal["deadzone", "sat"] = "deadzone",
        tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
    ) -> None:
        super().__init__(nz, nd, ne, optimizer, nonlinearity)
        self.tracker = tracker
        self.H = torch.nn.Parameter(torch.zeros((self.nz, self.nx)))

    def initialize_parameters(self) -> None:

        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, (C2 - H).T], [C2 - H, -2 * np.eye(self.nz)]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        M_gen = cp.bmat([[-np.eye(self.nz), H.T], [H, -X]])
        eps = 1e-3
        problem = cp.Problem(
            cp.Minimize([None]),
            [M << -eps * np.eye(nM), M_gen << -eps * np.eye(2 * self.nz)],
        )
        problem.solve(solver=self.optimizer)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")
        self.tracker.track(
            base_tracker.Log(
                "",
                f"Feasible initial parameter set found, write back parameters, problem status {problem.status}",
            )
        )

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)
        self.H.data = torch.tensor(H.value)

    def sdp_constraints(self) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            M11 = trans.torch_bmat(
                [
                    [-self.X, (self.C2 - self.H).T],
                    [self.C2 - self.H, -2 * torch.eye(self.nz)],
                ]
            )
            M21 = trans.torch_bmat([[self.A_tilde, self.B2_tilde]])
            M22 = -self.X
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        def general_sector() -> torch.Tensor:
            return -trans.torch_bmat(
                [[-torch.eye(self.nz), self.H.T], [self.H, -self.X]]
            )

        return [stability_lmi, general_sector]

    def project_parameters(self) -> None:
        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, (C2 - H).T], [C2 - H, -2 * np.eye(self.nz)]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        M0 = -self.sdp_constraints()[0]().detach().numpy()
        M_gen = cp.bmat([[-np.eye(self.nz), H.T], [H, -X]])
        M_gen0 = -self.sdp_constraints()[1]().detach().numpy()
        eps = 1e-3
        problem = cp.Problem(
            cp.Minimize(cp.norm(M - M0) + cp.norm(M_gen - M_gen0)),
            [M << -eps * np.eye(nM), M_gen << -eps * np.eye(2 * self.nz)],
        )
        problem.solve(solver=self.optimizer)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")
        self.tracker.track(
            base_tracker.Log(
                "",
                f"Project parameters to feasible set d: {np.linalg.norm(M0-M.value):.2f}, write back parameters, problem status {problem.status}",
            )
        )

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)
        self.H.data = torch.tensor(H.value)

    def check_constraints(self) -> bool:
        # check if constraints are psd
        with torch.no_grad():
            for lmi in self.sdp_constraints():
                _, info = torch.linalg.cholesky_ex(lmi())
                if info > 0:
                    return False
        return True
