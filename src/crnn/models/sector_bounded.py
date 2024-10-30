from . import base
import torch
import numpy as np
from typing import Union, Tuple, Callable, Literal, Optional, List
import cvxpy as cp
from .. import tracker as base_tracker
import numpy as np
from ..utils import transformation as trans
from numpy.typing import NDArray




class SectorBoundedLtiRnn(base.ConstrainedModule):
    def __init__(
            self, 
            nz:int, 
            nd: int,
            ne: int,
            optimizer: str = cp.MOSEK,
            nonlinearity:Literal['tanh', 'relu', 'deadzone']='tanh',
            tracker: Optional[base_tracker.BaseTracker] = base_tracker.BaseTracker()
    ) -> None:
        super().__init__(nz,nd,ne,optimizer,nonlinearity)
        self.tracker = tracker

    def sdp_constraints(self) -> List[Callable]:
        def stability_lmi() -> torch.Tensor:
            M11 = trans.torch_bmat(
                [
                    [-self.X, self.C2.T],
                    [self.C2, -2*torch.eye(self.nz)]
                ]
            )
            M21 = trans.torch_bmat(
                [
                    [self.A_tilde, self.B2_tilde]
                ]
            )
            M22 = -self.X
            M = trans.torch_bmat([
                [M11, M21.T],
                [M21, M22]
            ])
            return -M

        return [stability_lmi]


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
        problem.solve(solver=self.optimizer)
        if not problem.status == 'optimal':
            ValueError(f'cvxpy did not find a solution. {problem.status}')
        self.tracker.track(base_tracker.Log('', f'Feasible initial parameter set found, write back parameters, problem status {problem.status}'))

        self.A_tilde.data.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)

    def check_constraints(self)->bool:
        # check if constraints are psd
        with torch.no_grad():
            for lmi in self.sdp_constraints():
                _, info = torch.linalg.cholesky_ex(lmi())
                if info >0:
                    return False
        return True


        
    


class GeneralSectorBoundedLtiRnn(base.ConstrainedModule):
    def __init__(
            self, 
            nz:int, 
            nd: int,
            ne: int,
            optimizer: str = cp.MOSEK,
            nonlinearity:Literal['deadzone']='deadzone',
            tracker: Optional[base_tracker.BaseTracker] = base_tracker.BaseTracker()
    ) -> None:
        super().__init__(nz,nd,ne,optimizer,nonlinearity)
        self.tracker = tracker

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
        problem.solve(solver=self.optimizer)
        if not problem.status == 'optimal':
            ValueError(f'cvxpy did not find a solution. {problem.status}')
        self.tracker.track(base_tracker.Log('', f'Feasible initial parameter set found, write back parameters, problem status {problem.status}'))

        self.A_tilde.data.data = torch.tensor(A_tilde.value)
        self.B2_tilde.data = torch.tensor(B2_tilde.value)

        self.C2.data = torch.tensor(C2.value)
        self.X.data = torch.tensor(X.value)