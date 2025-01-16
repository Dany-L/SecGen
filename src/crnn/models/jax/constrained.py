from typing import Callable, List, Tuple

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import torch
from jax import Array, random
from jax.typing import ArrayLike

from ...utils import base as utils
from ...utils import transformation as trans
from . import base as base_jax
from .recurrent import BasicLtiRnn, get_model_matrices_from_flat_theta


class ConstrainedLtiRnn(BasicLtiRnn):
    def __init__(self, config: base_jax.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        self.parameter_names = [
            "A_tilde",
            "B_tilde",
            "B2_tilde",
            "C",
            "D",
            "D12",
            "C2_tilde",
            "D21_tilde",
            "D22",
            "X",
            "L",
        ]
        self.parameter_sizes = [
            (self.nx, self.nx),
            (self.nx, self.nd),
            (self.nx, self.nw),
            (self.ne, self.nx),
            (self.ne, self.nd),
            (self.ne, self.nw),
            (self.nz, self.nx),
            (self.nz, self.nd),
            (self.nz, self.nw),
            (self.nz, self.nz),
            (self.nz, 1),
        ]
        key = random.key(0)
        # self.theta = 1e-3 * random.normal(key, (self.get_number_of_parameters(),1))
        self.ga2 = 0.0
        self.theta = random.normal(key, (self.get_number_of_parameters(), 1))

    def initialize_parameters(self) -> str:

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
        ga2 = cp.Variable((1, 1))
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
        constraints = [M << -eps * np.eye(nM), *multiplier_constraints]
        problem = cp.Problem(cp.Minimize([None]), constraints)
        problem.solve(solver=self.sdp_opt)
        if not problem.status == "optimal":
            ValueError(f"cvxpy did not find a solution. {problem.status}")

        X = utils.get_opt_values(X)

        A_tilde = utils.get_opt_values(A_tilde)
        B_tilde = utils.get_opt_values(B_tilde)
        B2_tilde = utils.get_opt_values(B2_tilde)

        C = utils.get_opt_values(C)
        D = utils.get_opt_values(D)
        D12 = utils.get_opt_values(D12)

        L = self.set_L(utils.get_opt_values(L))

        C2_tilde = utils.get_opt_values(C2_tilde)
        D21_tilde = utils.get_opt_values(D21_tilde)
        D22 = np.zeros((self.nz, self.nw))

        theta_list = [
            A_tilde,
            B_tilde,
            B2_tilde,
            C,
            D,
            D12,
            C2_tilde,
            D21_tilde,
            D22,
            X,
            L,
        ]

        self.ga2 = 10 * ga2.value

        self.theta = np.vstack([p.flatten().reshape(-1, 1) for p in theta_list])

        return f"initialized theta with cvxpy solution, ga2: {ga2.value}, problem status: {problem.status}"

    def forward_unbatched(
        self, d: ArrayLike, x0: ArrayLike, theta: ArrayLike
    ) -> Tuple[Array, Array]:
        N, nu = d.shape  # number of batches, length of sequence, input size
        if x0 is None:
            x0 = jnp.zeros((self.nx,))
        else:
            x0 = x0[0]

        A, B, B2, C, D, D12, C2, D21, D22 = get_model_matrices_from_flat_theta(
            self, theta
        )

        ds = d.reshape((N, nu, 1))
        e_hats = []
        # e_hat = jnp.zeros((N, self.ne, 1))
        x_k = x0.reshape(self.nx, 1)

        for k in range(N):
            d_k = ds[k, :, :]
            w_k = self.nl(C2 @ x_k + D21 @ d_k)
            x_k = A @ x_k + B @ d_k + B2 @ w_k
            e_hats.append(C @ x_k + D @ d_k + D12 @ w_k)

            # e_hat = e_hat.at[k, :, :].set(e_hat_k)

        return (jnp.stack(e_hats).reshape(N, self.ne), x_k)

    def sdp_constraints(self, theta: ArrayLike) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            A_tilde, B_tilde, B2_tilde, C, D, D12, C2_tilde, D21_tilde, D22, X, L = (
                base_jax.get_matrices_from_flat_theta(self, theta)
            )
            L = self.get_L(L)
            ga2 = self.ga2

            M11 = trans.jax_bmat(
                [
                    [-X, np.zeros((self.nx, self.nd)), C2_tilde.T],
                    [np.zeros((self.nd, self.nx)), ga2 * np.eye(self.nd), D21_tilde.T],
                    [C2_tilde, D21_tilde, -2 * L],
                ]
            )
            M21 = trans.jax_bmat([[A_tilde, B_tilde, B2_tilde], [C, D, D12]])
            M22 = trans.jax_bmat(
                [
                    [-X, np.zeros((self.nx, self.ne))],
                    [np.zeros((self.ne, self.nx)), -np.eye(self.ne)],
                ]
            )
            M = trans.jax_bmat([[M11, M21.T], [M21, M22]])
            return -M

        return [stability_lmi]
