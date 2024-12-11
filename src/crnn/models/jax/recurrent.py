from typing import Callable, List, Tuple, Optional, Union

import jax.numpy as jnp
import numpy as np
import torch
from jax import Array, random, vmap
from jax.typing import ArrayLike

from . import base as base_jax


class BasicLtiRnn(base_jax.ConstrainedModule):
    CONFIG = base_jax.ConstrainedModuleConfig

    def __init__(self, config: base_jax.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        key = random.key(0)
        self.theta = 1e-3 * random.normal(key, (self.get_number_of_parameters(), 1))

    def forward_unbatched(
        self, d: ArrayLike, x0: ArrayLike, theta: ArrayLike
    ) -> Tuple[Array, Array]:
        N, nu = d.shape  # number of batches, length of sequence, input size
        if x0 is None:
            x0 = jnp.zeros((self.nx,))
        else:
            x0 = x0[0]
        A, B, B2, C, D, D12, C2, D21, D22 = get_matrices_from_flat_theta(self, theta)

        ds = d.reshape((N, nu, 1))
        e_hat = jnp.zeros((N, self.ne, 1))
        x_k = x0.reshape(self.nx, 1)

        for k in range(N):
            d_k = ds[k, :, :]
            w_k = self.nl(C2 @ x_k + D21 @ d_k)
            x_k = A @ x_k + B @ d_k + B2 @ w_k
            e_hat_k = C @ x_k + D @ d_k + D12 @ w_k

            e_hat = e_hat.at[k, :, :].set(e_hat_k)

        return (e_hat.reshape(N, self.ne), x_k)

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[ArrayLike] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        if isinstance(d, torch.Tensor):
            return_torch = True
            d = d.cpu().detach().numpy()
            if x0 is not None:
                x0 = [x0_i.cpu().detach().numpy() for x0_i in x0 if x0_i is not None]
        else:
            return_torch = False

        batch_forward = vmap(self.forward_unbatched, in_axes=(0, 0, None))
        e_hat, x = batch_forward(d, x0, theta)
        if return_torch:
            return (torch.from_dlpack(e_hat), torch.from_dlpack(x))
        else:
            return (e_hat, x)

    def sdp_constraints(self, theta: ArrayLike) -> List[Callable]:
        return [lambda: np.eye(1)]

    def pointwise_constraints(self) -> List[Callable]:
        return [lambda: np.eye(1)]

    def initialize_parameters(self) -> None:
        pass

    def project_parameters(self) -> None:
        pass

    def train(self) -> None:
        pass


def get_model_matrices_from_flat_theta(
    model: base_jax.ConstrainedModule, theta: ArrayLike
) -> List[Array]:
    A_tilde, B_tilde, B2_tilde, C, D, D12, C2_tilde, D21_tilde, D22, X, L = (
        get_matrices_from_flat_theta(model, theta)
    )

    X_inv = jnp.linalg.inv(X)
    A = X_inv @ A_tilde
    B = X_inv @ B_tilde
    B2 = X_inv @ B2_tilde
    # A = A_tilde
    # B2 = B2_tilde

    L_inv = jnp.linalg.inv(model.get_L(L))
    C2 = L_inv @ C2_tilde
    D21 = L_inv @ D21_tilde
    # C2, D21 = C2_tilde, D21_tilde

    return [A, B, B2, C, D, D12, C2, D21, D22]


def get_matrices_from_flat_theta(
    model: base_jax.ConstrainedModule, theta: ArrayLike
) -> List[Array]:
    start_flat = 0
    params = []
    for par_size in model.parameter_sizes:
        num_par = np.prod(par_size)
        params.append(theta[start_flat : start_flat + num_par].reshape(par_size))
        start_flat += num_par

    return params
