import torch
from typing import List
import numpy as np
from numpy.typing import NDArray
from ..configuration import LureSystemClass

def get_lure_matrices(
    gen_plant: torch.Tensor,
    nx: int, #state
    nd:int, # input
    ne:int, # output
    nonlinearity: torch.nn
) -> LureSystemClass:
        A = gen_plant[: nx, : nx]
        B = gen_plant[:nx, nx:nx+nd]
        B2 = gen_plant[:nx, nx+nd:]

        C = gen_plant[nx:nx+ne, : nx]
        D = gen_plant[nx:nx+ne, nx:nx+nd]
        D12 = gen_plant[nx:nx+ne, nx+nd:]

        C2 = gen_plant[nx+ne:, : nx]
        D21 = gen_plant[nx+ne:, nx:nx+nd]
        D22 = gen_plant[nx+ne:, nx+nd:]

        return LureSystemClass(
                A,B,B2,C,D,D12,D21,D22,nonlinearity
        )

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
