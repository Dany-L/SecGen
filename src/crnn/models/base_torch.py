from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Type, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from pydantic import BaseModel
from . import base

from ..utils import transformation as trans


class DynamicIdentificationConfig(base.DynamicIdentificationConfig):
    pass


class DynamicIdentificationModel(base.DynamicIdentificationModel, nn.Module):
    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConstrainedModuleConfig(DynamicIdentificationConfig):
    sdp_opt: str = cp.MOSEK
    nonlinearity: Literal["tanh", "relu", "deadzone", "sat"] = "tanh"
    multiplier: Literal["none", "diag", "zf"] = "none"


class ConstrainedModule(DynamicIdentificationModel):
    def __init__(self, config: ConstrainedModuleConfig) -> None:
        super().__init__(config)

        self.nl: Optional[nn.Module] = None
        if config.nonlinearity == "tanh":
            self.nl = nn.Tanh()
        elif config.nonlinearity == "relu":
            self.nl = nn.ReLU()
        elif config.nonlinearity == "deadzone":
            self.nl = nn.Softshrink()
        elif config.nonlinearity == "sat":
            self.nl = nn.Hardtanh()
        else:
            raise ValueError(f"Unsupported nonlinearity: {config.nonlinearity}")

        self.sdp_opt = config.sdp_opt

        self.multiplier = config.multiplier
        if self.multiplier == "none":
            self.L = torch.eye(self.nz)
        elif config.multiplier == "diag":
            self.L = torch.nn.Parameter(torch.ones((self.nz,)))
        else:
            raise NotImplementedError(f"Unsupported multiplier: {config.multiplier}")

    def get_optimization_multiplier_and_constraints(
        self,
    ) -> Tuple[Union[NDArray[np.float64], cp.Variable], List[cp.Constraint]]:
        if self.multiplier == "none":
            L = np.eye(self.nz)
            multiplier_constraints = []
        elif self.multiplier == "diag":
            la = cp.Variable((self.nz,))
            L = cp.diag(la)
            multiplier_constraints = [lam >= 0 for lam in la]
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")
        return (L, multiplier_constraints)

    def get_L(self) -> torch.Tensor:
        if self.multiplier == "none":
            return self.L
        elif self.multiplier == "diag":
            return torch.diag(self.L)
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")

    def set_L(self, L: torch.Tensor) -> None:
        if self.multiplier == "none":
            self.L = L
        elif self.multiplier == "diag":
            self.L.value = torch.diag(L)
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")

    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[List] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        assert self.lure._nu == nu
        if x0 is None:
            x0 = torch.zeros(size=(B, self.nx))
        else:
            x0 = x0[0]
        ds = d.reshape(shape=(B, N, nu, 1))
        es_hat, x = self.lure.forward(x0=x0, us=ds, return_states=True)
        return (
            es_hat.reshape(B, N, self.lure._ny),
            (x.reshape(B, self.nx),),
        )

    def check_constraints(self) -> bool:
        # check if constraints are psd
        with torch.no_grad():
            for lmi in self.sdp_constraints():
                _, info = torch.linalg.cholesky_ex(lmi())
                if info > 0:
                    return False

            for scalar in self.pointwise_constraints():
                if scalar() < 0:
                    return False
        return True

    def get_phi(self, t: float) -> torch.Tensor:
        if self.sdp_constraints() is not None:
            batch_phi = (
                1
                / t
                * torch.sum(
                    torch.tensor([-torch.logdet(M()) for M in self.sdp_constraints()])
                )
            )
        else:
            batch_phi = torch.tensor(0.0)
        return batch_phi


class StableConstrainedModule(ConstrainedModule):
    def __init__(self, config: ConstrainedModuleConfig) -> None:
        super().__init__(config)
        mean = 0.0
        self.A_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))
        self.B = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nx, self.nd)),
                1 / self.nx * torch.ones((self.nx, self.nd)),
            )
        )
        self.B2_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nw)))

        self.C = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nx)),
                1 / self.ne * torch.ones((self.ne, self.nx)),
            )
        )
        self.D = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nd)),
                1 / self.ne * torch.ones((self.ne, self.nd)),
            )
        )
        self.D12 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nw)),
                1 / self.ne * torch.ones((self.ne, self.nw)),
            )
        )

        self.C2_tilde = torch.nn.Parameter(torch.zeros((self.nz, self.nx)))
        self.D21 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nd)),
                1 / self.nz * torch.ones((self.nz, self.nd)),
            )
        )
        self.D22 = torch.zeros((self.nz, self.nw))

        self.X = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))

    def set_lure_system(self) -> base.LureSystemClass:
        X_inv = torch.linalg.inv(self.X)
        A = X_inv @ self.A_tilde
        B2 = X_inv @ self.B2_tilde

        C2 = torch.linalg.inv(self.get_L()) @ self.C2_tilde

        theta = trans.torch_bmat(
            [[A, self.B, B2], [self.C, self.D, self.D12], [C2, self.D21, self.D22]]
        )
        sys = base.get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = base.LureSystem(sys)

        return sys


class IoConstrainedModule(ConstrainedModule):
    def __init__(self, config: ConstrainedModuleConfig) -> None:
        super().__init__(config)
        mean = 0.0
        self.A = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nx, self.nx)),
                1 / self.nx * torch.ones((self.nx, self.nx)),
            )
        )
        self.B = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nx, self.nd)),
                1 / self.nx * torch.ones((self.nx, self.nd)),
            )
        )
        self.B2 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nx, self.nw)),
                1 / self.nx * torch.ones((self.nx, self.nw)),
            )
        )

        self.C = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nx)),
                1 / self.ne * torch.ones((self.ne, self.nx)),
            )
        )
        self.D = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nd)),
                1 / self.ne * torch.ones((self.ne, self.nd)),
            )
        )
        self.D12 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nw)),
                1 / self.ne * torch.ones((self.ne, self.nw)),
            )
        )

        self.C2 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nx)),
                1 / self.ne * torch.ones((self.nz, self.nx)),
            )
        )
        self.D21_tilde = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nd)),
                1 / self.nz * torch.ones((self.nz, self.nd)),
            )
        )
        self.D22 = torch.zeros((self.nz, self.nw))

        self.ga2 = torch.nn.Parameter(torch.ones((1, 1)))

    def set_lure_system(self) -> base.LureSystemClass:

        D21 = torch.linalg.inv(self.get_L()) @ self.D21_tilde

        theta = trans.torch_bmat(
            [
                [self.A, self.B, self.B2],
                [self.C, self.D, self.D12],
                [self.C2, D21, self.D22],
            ]
        )
        sys = base.get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = base.LureSystem(sys)

        return sys


class OptFcn:
    def __init__(
        self,
        d: torch.Tensor,
        e: torch.Tensor,
        nn: ConstrainedModule,
        t=float,
        x0: torch.Tensor = None,
        l: torch.nn.Module = torch.nn.MSELoss(),
    ) -> None:
        self.d = d
        self.x0 = x0
        self.e = e
        self.nn = nn
        self.l = l
        self.t = torch.tensor(t)

    def f(self, theta: torch.Tensor) -> torch.Tensor:
        self.set_vec_pars_to_model(theta)
        self.nn.set_lure_system()
        l = self.l(self.nn(self.d, self.x0)[0], self.e)
        phi = self.nn.get_phi(self.t)
        return l + phi

    def dF(self, theta: torch.Tensor) -> torch.Tensor:
        self.nn.zero_grad()
        loss = self.f(theta)
        loss.backward(retain_graph=True)
        grads = []
        for p in self.nn.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
            else:
                grads.append(torch.zeros_like(p).flatten())

        return torch.hstack(grads).reshape(-1, 1)

    def set_vec_pars_to_model(self, theta: torch.Tensor) -> None:
        start_flat = 0
        for p in self.nn.parameters():
            num_par = p.numel()
            p.data = theta[start_flat : start_flat + num_par].view(p.shape)
            start_flat += num_par
