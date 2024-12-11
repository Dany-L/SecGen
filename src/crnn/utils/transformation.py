from typing import List

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
