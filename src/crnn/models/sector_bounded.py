from typing import Callable, List

import cvxpy as cp
import numpy as np
import torch

from ..utils import base as utils
from ..utils import transformation as trans
from . import base


class SectorBoundedLtiRnn(base.StableConstrainedModule):
    CONFIG = base.ConstrainedModuleConfig

    def __init__(self, config: base.ConstrainedModuleConfig) -> None:
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
            M11 = trans.torch_bmat(
                [[-self.X, self.C2_tilde.T], [self.C2_tilde, -2 * L]]
            )
            M21 = trans.torch_bmat([[self.A_tilde, self.B2_tilde]])
            M22 = -self.X
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        return [stability_lmi]

    def initialize_parameters(self) -> str:

        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()

        C2_tilde = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, C2_tilde.T], [C2_tilde, -2 * L]])
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

        self.A_tilde.data = torch.tensor(utils.get_opt_values(A_tilde))
        self.B2_tilde.data = torch.tensor(utils.get_opt_values(B2_tilde))

        self.C2_tilde.data = torch.tensor(utils.get_opt_values(C2_tilde))
        self.X.data = torch.tensor(utils.get_opt_values(X))
        self.set_L(torch.tensor(utils.get_opt_values(L)))

        return problem.status

    def project_parameters(self) -> str:
        X = cp.Variable((self.nx, self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()

        C2_tilde = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, C2_tilde.T], [C2_tilde, -2 * L]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        M0 = -self.sdp_constraints()[0]().cpu().detach().numpy()
        problem = cp.Problem(
            cp.Minimize(cp.norm(M0 - M)),
            [M << -eps * np.eye(nM), *multiplier_constraints],
        )
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.A_tilde.data = torch.tensor(utils.get_opt_values(A_tilde))
        self.B2_tilde.data = torch.tensor(utils.get_opt_values(B2_tilde))

        self.set_L(torch.tensor(utils.get_opt_values(L)))

        self.C2_tilde.data = torch.tensor(utils.get_opt_values(C2_tilde))
        self.X.data = torch.tensor(utils.get_opt_values(X))

        return problem.status


class GeneralSectorBoundedLtiRnn(base.StableConstrainedModule):
    CONFIG = base.ConstrainedModuleConfig

    def __init__(self, config: base.ConstrainedModuleConfig) -> None:
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
            M11 = trans.torch_bmat(
                [
                    [-self.X, (self.C2_tilde - self.H).T],
                    [self.C2_tilde - self.H, -2 * torch.eye(self.nz)],
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

    def initialize_parameters(self) -> str:

        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2_tilde = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, (C2_tilde - H).T], [C2_tilde - H, -2 * L]])
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

        self.A_tilde.data = torch.tensor(utils.get_opt_values(A_tilde))
        self.B2_tilde.data = torch.tensor(utils.get_opt_values(B2_tilde))

        self.set_L(torch.tensor(utils.get_opt_values(L)))

        self.C2_tilde.data = torch.tensor(utils.get_opt_values(C2_tilde))
        self.X.data = torch.tensor(utils.get_opt_values(X))
        self.H.data = torch.tensor(utils.get_opt_values(H))

        return problem.status

    def project_parameters(self) -> str:
        X = cp.Variable((self.nx, self.nx), symmetric=True)
        H = cp.Variable((self.nz, self.nx))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()

        A_tilde = cp.Variable((self.nx, self.nx))
        B2_tilde = cp.Variable((self.nx, self.nw))

        C2_tilde = cp.Variable((self.nz, self.nx))

        M11 = cp.bmat([[-X, (C2_tilde - H).T], [C2_tilde - H, -2 * L]])
        M21 = cp.bmat([[A_tilde, B2_tilde]])
        M22 = -X
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        M0 = -self.sdp_constraints()[0]().cpu().detach().numpy()
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
        self.X.data = torch.tensor(utils.get_opt_values(X))
        self.H.data = torch.tensor(utils.get_opt_values(H))
        self.set_L(torch.tensor(utils.get_opt_values(L)))

        return problem.status


class BasicLtiRnn(base.ConstrainedModule):
    CONFIG = base.ConstrainedModuleConfig

    def __init__(self, config: base.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        self.nonlinearity = config.nonlinearity

        mean = 0.0
        self.A = torch.zeros((self.nx, self.nx))
        self.B = torch.zeros((self.nx, self.nd))
        self.B2 = torch.eye(self.nw)

        self.C = torch.zeros((self.ne, self.nx))
        self.D = torch.zeros((self.ne, self.nd))

        self.D12 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nw)),
                1 / self.ne * torch.ones((self.ne, self.nw)),
            )
        )
        self.C2 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nx)),
                1 / self.nz * torch.ones((self.nz, self.nx)),
            )
        )
        self.D21 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nd)),
                1 / self.nz * torch.ones((self.nz, self.nd)),
            )
        )
        self.D22 = torch.zeros((self.nz, self.nw))

    def set_lure_system(self) -> base.LureSystemClass:
        theta = trans.torch_bmat(
            [
                [self.A, self.B, self.B2],
                [self.C, self.D, self.D12],
                [self.C2, self.D21, self.D22],
            ]
        )
        sys = base.get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = base.LureSystem(sys)

        return sys

    def check_constraints(self) -> bool:
        return True

    def sdp_constraints(self) -> List[Callable]:
        pass

    def initialize_parameters(self) -> None:
        pass

    def project_parameters(self) -> None:
        pass
