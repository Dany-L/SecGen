from . import base
import torch
import numpy as np
from typing import Union, Tuple, Callable, Literal
import cvxpy as cp



class SectorBoundedLtiRnn(base.ConstrainedModule):
    def __init__(
            self, 
            nz:int, 
            nd: int,
            ne: int,
            device: torch.device=torch.device('cpu'),
            optimizer: str = cp.MOSEK,
            nonlinearity:Literal['tanh', 'relu', 'deadzone']='tanh'
    ) -> None:
        super().__init__(nz,nd,ne,device,optimizer,nonlinearity)

    def initialize_parameters(self) -> None:

        X = cp.Variable((self.nx,self.nx), symmetric=True)

        A_tilde = cp.Variable((self.nx,self.nx))
        B = cp.Variable((self.nx,self.nd))
        B2_tilde = cp.Variable((self.nx,self.nw))
        
        C = cp.Variable((self.ne,self.nx))
        D = cp.Variable((self.ne,self.nd))
        D12 = cp.Variable((self.ne,self.nw))

        C2 = cp.Variable((self.nz,self.nx))
        D21 = cp.Variable((self.nz,self.nd))
        
        M11 = cp.bmat(
            [
                [-X, C2.T],
                [C2, -2*np.eye(self.nz)]
            ]
        )
        M21 = cp.bmat(
            [
                [A_tilde, B2_tilde]
            ]
        )
        M22 = -X
        M = cp.bmat([
            [M11, M21.T],
            [M21, M22]
        ])
        nM = M.shape[0]
        eps = 1e-3
        problem = cp.Problem(cp.Minimize([None]),[M<< -eps * np.eye(nM)])
        problem.solve(solver=self.optimizer, verbose=True)
        if not problem.status == 'optimal':
            ValueError(f'cvxpy did not find a solution. {problem.status}')

        self.A_tilde.data.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)

    def check_constraints(self)->bool:
        return True
        
    


class GeneralSectorBoundedLtiRnn(base.ConstrainedModule):
    def __init__(self, nz, nonlinearity = 'tanh'):
        super().__init__(nz, nonlinearity)