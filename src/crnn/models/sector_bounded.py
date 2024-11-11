from typing import Callable, List

import cvxpy as cp
import numpy as np
import torch

from ..utils import transformation as trans
from . import base


class SectorBoundedLtiRnn(base.ConstrainedModule):
    CONFIG = base.ConstrainedModuleConfig

    def __init__(self, config: base.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        # self.tracker = tracker

    def pointwise_constraints(self) -> List[Callable]:
        constraint_fcn: List[Callable] = []
        for n_k in range(self.nz):
            constraint_fcn.append(lambda: self.la[n_k])
        return constraint_fcn

    def sdp_constraints(self) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            L = torch.diag(self.la)
            M11 = trans.torch_bmat([[-self.X, self.C2.T], [self.C2, -2 * L]])
            M21 = trans.torch_bmat([[self.A_tilde, self.B2_tilde]])
            M22 = -self.X
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        return [stability_lmi]

    def initialize_parameters(self) -> str:

        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        la = cp.Variable((self.nz,))
        L = cp.diag(la)
        multiplier_constraints = [lam >= 0 for lam in la]

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, C2.T], [C2, -2 * L]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        constraints = [M << -eps * np.eye(nM), *multiplier_constraints]
        problem = cp.Problem(cp.Minimize([None]), constraints)
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)
        self.la.data = torch.tensor(la.value)

        return problem.status

    def project_parameters(self) -> str:
        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        la = cp.Variable((self.nz,))
        L = cp.diag(la)
        multiplier_constraints = [lam >= 0 for lam in la]

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, C2.T], [C2, -2 * L]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        M0 = -self.sdp_constraints()[0]().detach().numpy()
        problem = cp.Problem(
            cp.Minimize(cp.norm(M0 - M)),
            [M << -eps * np.eye(nM), *multiplier_constraints],
        )
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.la.data = torch.tensor(la.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)

        return problem.status


class GeneralSectorBoundedLtiRnn(base.ConstrainedModule):
    CONFIG = base.ConstrainedModuleConfig

    def __init__(self, config: base.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        # self.tracker = tracker
        self.H = torch.nn.Parameter(torch.zeros((self.nz, self.nx)))

    def pointwise_constraints(self) -> List[Callable]:
        constraint_fcn: List[Callable] = []
        for n_k in range(self.nz):
            constraint_fcn.append(lambda: self.la[n_k])
        return constraint_fcn

    def initialize_parameters(self) -> str:

        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        la = cp.Variable((self.nz,))
        L = cp.diag(la)
        multiplier_constraints = [lam >= 0 for lam in la]

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, (C2 - H).T], [C2 - H, -2 * L]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        M_gen = cp.bmat([[-np.eye(self.nz), H.T], [H, -X]])
        eps = 1e-3

        constraints = [
            M << -eps * np.eye(nM),
            M_gen << -eps * np.eye(2 * self.nz),
            *multiplier_constraints,
        ]
        problem = cp.Problem(cp.Minimize([None]), constraints)
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)
        self.H.data = torch.tensor(H.value)

        return problem.status

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

    def project_parameters(self) -> str:
        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        la = cp.Variable((self.nz,))
        L = cp.diag(la)
        multiplier_constraints = [lam >= 0 for lam in la]

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2 = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, (C2 - H).T], [C2 - H, -2 * L]])
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
            [
                M << -eps * np.eye(nM),
                M_gen << -eps * np.eye(2 * self.nz),
                *multiplier_constraints,
            ],
        )
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.A_tilde.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)
        self.H.data = torch.tensor(H.value)
        self.la.data = torch.tensor(la.value)

        return problem.status
