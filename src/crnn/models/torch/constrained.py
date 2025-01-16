from typing import Callable, List

import cvxpy as cp
import numpy as np
import torch

from ...utils import base as utils
from ...utils import transformation as trans
from . import base as base_torch


class ConstrainedLtiRnn(base_torch.L2StableConstrainedModule):
    CONFIG = base_torch.ConstrainedModuleConfig

    def __init__(self, config: base_torch.ConstrainedModuleConfig) -> None:
        super().__init__(config)

    def pointwise_constraints(self) -> List[Callable]:
        constraint_fcn: List[Callable] = []
        la = torch.diag(self.get_L())
        for n_k in range(self.nz):
            constraint_fcn.append(lambda: la[n_k])
        return constraint_fcn

    def sdp_constraints(self) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            L = self.get_L()
            X = self.get_X()
            M11 = trans.torch_bmat(
                [
                    [-X, torch.zeros((self.nx, self.nd)), self.C2_tilde.T],
                    [
                        torch.zeros((self.nd, self.nx)),
                        -self.ga2 * torch.eye(self.nd),
                        self.D21_tilde.T,
                    ],
                    [self.C2_tilde, self.D21_tilde, -2 * L],
                ]
            )
            M21 = trans.torch_bmat(
                [
                    [self.A_tilde, self.B_tilde, self.B2_tilde],
                    [self.C, self.D, self.D12],
                ]
            )
            M22 = trans.torch_bmat(
                [
                    [-X, torch.zeros((self.nx, self.ne))],
                    [torch.zeros((self.ne, self.nx)), -torch.eye(self.ne)],
                ]
            )
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        return [stability_lmi]

    def initialize_parameters(self) -> str:
        return self.project_parameters()

    def project_parameters(self) -> str:
        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B_tilde = cp.Variable((self.nx, self.nd))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C = cp.Variable((self.ne, self.nx))
        D = cp.Variable((self.ne, self.nd))
        D12 = cp.Variable((self.ne, self.nw))

        C2_tilde = cp.Variable((self.nz, self.nx))
        D21_tilde = cp.Variable((self.nz, self.nd))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()
        ga2 = self.ga2.data
        # ga2 = cp.Variable((1, 1))
        if self.nd == 1:
            M11_22 = -ga2
        else:
            M11_22 = -ga2 * np.eye(self.nd)
        M11 = cp.bmat(
            [
                [-X, np.zeros((self.nx, self.nd)), C2_tilde.T],
                [np.zeros((self.nd, self.nx)), M11_22, D21_tilde.T],
                [C2_tilde, D21_tilde, -2 * L],
            ]
        )
        M21 = cp.bmat([[A_tilde, B_tilde, B2_tilde], [C, D, D12]])
        M22 = cp.bmat(
            [
                [-X, np.zeros((self.nx, self.ne))],
                [np.zeros((self.ne, self.nx)), -np.eye(self.ne)],
            ]
        )
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        M0 = self.sdp_constraints()[0]().cpu().detach().numpy()
        problem = cp.Problem(
            cp.Minimize(cp.norm(M0 - M)),
            # cp.Minimize([None]),
            [M << -eps * np.eye(nM), *multiplier_constraints],
        )
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.A_tilde.data = torch.tensor(utils.get_opt_values(A_tilde))
        self.B_tilde.data = torch.tensor(utils.get_opt_values(B_tilde))
        self.B2_tilde.data = torch.tensor(utils.get_opt_values(B2_tilde))

        self.C.data = torch.tensor(utils.get_opt_values(C))
        self.D.data = torch.tensor(utils.get_opt_values(D))
        self.D12.data = torch.tensor(utils.get_opt_values(D12))

        self.set_L(torch.tensor(utils.get_opt_values(L)))

        self.C2_tilde.data = torch.tensor(utils.get_opt_values(C2_tilde))
        self.D21_tilde.data = torch.tensor(utils.get_opt_values(D21_tilde))

        self.Lx.data = torch.tensor(np.linalg.cholesky(utils.get_opt_values(X)))
        # self.X.data = torch.tensor(utils.get_opt_values(X))

        return f"Projected parameters with cvxpy solution, ga2: {utils.get_opt_values(ga2)}, problem status: {problem.status}"


class ConstrainedLtiRnnGeneralSectorConditions(base_torch.L2StableConstrainedModule):
    CONFIG = base_torch.ConstrainedModuleConfig

    def __init__(self, config: base_torch.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        # self.tracker = tracker
        self.H = torch.nn.Parameter(torch.zeros((self.nz, self.nx)))

    def pointwise_constraints(self) -> List[Callable]:
        constraint_fcn: List[Callable] = []
        la = torch.diag(self.get_L())
        for n_k in range(self.nz):
            constraint_fcn.append(lambda: la[n_k])
        return constraint_fcn

    def sdp_constraints(self) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            L = self.get_L()
            X = self.get_X()
            M11 = trans.torch_bmat(
                [
                    [-X, torch.zeros((self.nx, self.nd)), (self.C2_tilde - self.H).T],
                    [
                        torch.zeros((self.nd, self.nx)),
                        -self.ga2 * torch.eye(self.nd),
                        self.D21_tilde.T,
                    ],
                    [self.C2_tilde - self.H, self.D21_tilde, -2 * L],
                ]
            )
            M21 = trans.torch_bmat(
                [
                    [self.A_tilde, self.B_tilde, self.B2_tilde],
                    [self.C, self.D, self.D12],
                ]
            )
            M22 = trans.torch_bmat(
                [
                    [-X, torch.zeros((self.nx, self.ne))],
                    [torch.zeros((self.ne, self.nx)), -torch.eye(self.ne)],
                ]
            )
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        def general_sector() -> torch.Tensor:
            X = self.get_X()
            return -trans.torch_bmat([[-torch.eye(self.nz), self.H.T], [self.H, -X]])

        return [stability_lmi, general_sector]

    def initialize_parameters(self) -> str:
        return self.project_parameters()

    def project_parameters(self) -> str:
        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        A_tilde = cp.Variable((self.nx, self.nx))
        B_tilde = cp.Variable((self.nx, self.nd))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C = cp.Variable((self.ne, self.nx))
        D = cp.Variable((self.ne, self.nd))
        D12 = cp.Variable((self.ne, self.nw))

        C2_tilde = cp.Variable((self.nz, self.nx))
        D21_tilde = cp.Variable((self.nz, self.nd))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()
        ga2 = self.ga2.data
        # ga2 = cp.Variable((1, 1))
        if self.nd == 1:
            M11_22 = -ga2
        else:
            M11_22 = -ga2 * np.eye(self.nd)
        M11 = cp.bmat(
            [
                [-X, np.zeros((self.nx, self.nd)), (C2_tilde - H).T],
                [np.zeros((self.nd, self.nx)), M11_22, D21_tilde.T],
                [C2_tilde - H, D21_tilde, -2 * L],
            ]
        )
        M21 = cp.bmat([[A_tilde, B_tilde, B2_tilde], [C, D, D12]])
        M22 = cp.bmat(
            [
                [-X, np.zeros((self.nx, self.ne))],
                [np.zeros((self.ne, self.nx)), -np.eye(self.ne)],
            ]
        )
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        M0 = -self.sdp_constraints()[0]().cpu().detach().numpy()
        nM = M.shape[0]

        M_gen = cp.bmat([[-np.eye(self.nz), H.T], [H, -X]])
        M_gen0 = -self.sdp_constraints()[1]().cpu().detach().numpy()

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

        self.A_tilde.data = torch.tensor(utils.get_opt_values(A_tilde))
        self.B2_tilde.data = torch.tensor(utils.get_opt_values(B2_tilde))

        self.C2_tilde.data = torch.tensor(utils.get_opt_values(C2_tilde))
        self.Lx.data = torch.tensor(np.linalg.cholesky(utils.get_opt_values(X)))
        self.H.data = torch.tensor(utils.get_opt_values(H))
        self.set_L(torch.tensor(utils.get_opt_values(L)))

        return f"Projected parameters with cvxpy solution, ga2: {utils.get_opt_values(ga2)}, problem status: {problem.status}"
