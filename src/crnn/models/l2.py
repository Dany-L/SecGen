from typing import Callable, List

import cvxpy as cp
import numpy as np
import torch

from ..utils import base as utils
from ..utils import transformation as trans
from . import base_torch


class L2Stable(base_torch.IoConstrainedModule):
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
            M11 = trans.torch_bmat(
                [[-self.ga2, self.D21_tilde.T], [self.D21_tilde, -2 * L]]
            )
            M21 = trans.torch_bmat([[self.D, self.D12]])
            M22 = -torch.eye(self.ne)
            M = trans.torch_bmat([[M11, M21.T], [M21, M22]])
            return -M

        return [stability_lmi]

    def initialize_parameters(self) -> str:

        D = cp.Variable((self.ne, self.nd))
        D12 = cp.Variable((self.ne, self.nw))

        D21_tilde = cp.Variable((self.nz, self.nd))
        ga2 = cp.Variable((1, 1))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()

        M11 = cp.bmat([[-ga2 * np.eye(self.nd), D21_tilde.T], [D21_tilde, -2 * L]])
        M21 = cp.bmat([[D, D12]])
        M22 = -np.eye(self.ne)
        M = cp.bmat([[M11, M21.T], [M21, M22]])
        nM = M.shape[0]
        eps = 1e-3
        constraints = [M << -eps * np.eye(nM), *multiplier_constraints]
        problem = cp.Problem(cp.Minimize([None]), constraints)
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        self.D.data = torch.tensor(utils.get_opt_values(D))
        self.D12.data = torch.tensor(utils.get_opt_values(D12))

        self.D21_tilde.data = torch.tensor(utils.get_opt_values(D21_tilde))
        self.set_L(torch.tensor(utils.get_opt_values(L)))
        self.ga2.data = torch.tensor(utils.get_opt_values(ga2))

        return problem.status

    def project_parameters(self) -> str:
        ga2 = cp.Variable((1, 1))

        D = cp.Variable((self.ne, self.nd))
        D12 = cp.Variable((self.ne, self.nw))

        D21_tilde = cp.Variable((self.nz, self.nd))

        L, multiplier_constraints = self.get_optimization_multiplier_and_constraints()

        M11 = cp.bmat([[-ga2 * np.eye(self.nd), D21_tilde.T], [D21_tilde, -2 * L]])
        M21 = cp.bmat([[D, D12]])
        M22 = -np.eye(self.ne)
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

        self.D.data = torch.tensor(utils.get_opt_values(D))
        self.D12.data = torch.tensor(utils.get_opt_values(D12))

        self.D21_tilde.data = torch.tensor(utils.get_opt_values(D21_tilde))
        self.ga2.data = torch.tensor(utils.get_opt_values(ga2))
        self.set_L(torch.tensor(utils.get_opt_values(L)))

        return problem.status
