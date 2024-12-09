from typing import Callable, List, Union, Tuple

import cvxpy as cp
import numpy as np
import jax.numpy as jnp
from jax import Array, vmap
from jax.typing import ArrayLike
from jax import random
import torch

from ..utils import base as utils
from ..utils import transformation as trans
from . import base_jax


class BasicLtiRnn(base_jax.ConstrainedModule):
    CONFIG = base_jax.ConstrainedModuleConfig

    def __init__(self, config: base_jax.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        key = random.key(0)
        n_outs = [self.nx, self.ne, self.nz]
        self.theta = base_jax.init_network_params(
            n_outs, key, self.nx, self.nd, self.nw
        )

    def forward_unbatched(
        self, d: ArrayLike, x0: ArrayLike, theta: ArrayLike
    ) -> Tuple[Array, Array]:
        N, nu = d.shape  # number of batches, length of sequence, input size
        if x0 is None:
            x0 = jnp.zeros((self.nx,))
        else:
            x0 = x0[0]
        ds = d.reshape((N, nu, 1))
        e_hat = jnp.zeros((N, self.ne, 1))
        x_k = x0.reshape(self.nx, 1)

        for k in range(N):
            d_k = ds[k, :, :]
            for x_param, d_param, _ in theta[2:]:
                w_k = self.nl(x_param @ x_k + d_param @ d_k)

            x_k = theta[0][0] @ x_k + theta[0][1] @ d_k + theta[0][2] @ w_k
            e_hat_k = theta[1][0] @ x_k + theta[1][1] @ d_k + theta[1][2] @ w_k

            e_hat = e_hat.at[k, :, :].set(e_hat_k)

        return (e_hat.reshape(N, self.ne), x_k)

    def forward(
        self, d: torch.Tensor, x0: torch.Tensor, theta: List
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def check_constraints(self) -> bool:
        return True

    def sdp_constraints(self) -> List[Callable]:
        pass

    def initialize_parameters(self) -> None:
        pass

    def project_parameters(self) -> None:
        pass

    def train(self) -> None:
        pass
