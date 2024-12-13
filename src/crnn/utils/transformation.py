from typing import List, Iterator

import jax.numpy as jnp
import numpy as np
import torch
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray


def bmat(mat: List[List[NDArray[np.float64]]]) -> NDArray[np.float64]:
    mat_list = []
    for col in mat:
        mat_list.append(np.hstack(col))
    return np.vstack(mat_list)


def torch_bmat(mat: List[List[torch.Tensor]]) -> torch.Tensor:
    mat_list = []
    for col in mat:
        mat_list.append(torch.hstack(col))
    return torch.vstack(mat_list)


def jax_bmat(mat: List[List[ArrayLike]]) -> Array:
    mat_list = []
    for col in mat:
        mat_list.append(jnp.hstack(col))
    return jnp.vstack(mat_list)

def get_flat_parameters(params: Iterator[torch.nn.parameter.Parameter]) -> torch.Tensor:
    return torch.hstack(
        [
            p.flatten().clone()
            for p in params
            if p is not None
        ]
    ).reshape(-1, 1)

def set_vec_pars_to_model(params: Iterator[torch.nn.parameter.Parameter], theta: torch.Tensor) -> None:
    start_flat = 0
    for p in params:
        num_par = p.numel()
        p.data = theta[start_flat : start_flat + num_par].view(p.shape)
        start_flat += num_par
