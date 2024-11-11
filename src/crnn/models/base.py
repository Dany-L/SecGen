from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from ..utils import transformation as trans

# from .. import tracker as base_tracker


@dataclass
class LureSystemClass:
    A: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    D12: torch.Tensor
    C2: torch.Tensor
    D21: torch.Tensor
    D22: torch.Tensor
    Delta: torch.nn.Module


class DynamicIdentificationConfig(BaseModel):
    nd: int
    ne: int
    nz: int


def get_lure_matrices(
    gen_plant: torch.Tensor,
    nx: int,  # state
    nd: int,  # input
    ne: int,  # output
    nonlinearity: Optional[torch.nn.Module] = torch.nn.Tanh(),
) -> LureSystemClass:
    A = gen_plant[:nx, :nx]
    B = gen_plant[:nx, nx : nx + nd]
    B2 = gen_plant[:nx, nx + nd :]

    C = gen_plant[nx : nx + ne, :nx]
    D = gen_plant[nx : nx + ne, nx : nx + nd]
    D12 = gen_plant[nx : nx + ne, nx + nd :]

    C2 = gen_plant[nx + ne :, :nx]
    D21 = gen_plant[nx + ne :, nx : nx + nd]
    D22 = gen_plant[nx + ne :, nx + nd :]

    return LureSystemClass(A, B, B2, C, D, D12, C2, D21, D22, nonlinearity)


class DynamicIdentificationModel(nn.Module, ABC):

    def __init__(self, config: DynamicIdentificationConfig) -> None:
        super().__init__()
        self.nz, self.nw = config.nz, config.nz  # size of nonlinear channel
        # internal state size matches size of nonlinear channel for this network
        self.nx = self.nz
        self.nd, self.ne = config.nd, config.ne  # size of input and output

    @abstractmethod
    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    @abstractmethod
    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    @abstractmethod
    def initialize_parameters(self) -> str:
        pass

    def set_lure_system(self) -> LureSystemClass:
        return LureSystem(
            get_lure_matrices(
                torch.zeros(
                    size=(self.nx + self.ne + self.nw, self.nx + self.nd + self.nz)
                ),
                self.nx,
                self.nd,
                self.ne,
            )
        )

    @abstractmethod
    def forward(
        self, d: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def check_constraints(self) -> bool:
        return True

    @abstractmethod
    def project_parameters(self) -> str:
        pass


class ConstrainedModuleConfig(DynamicIdentificationConfig):
    sdp_opt: str = cp.MOSEK
    nonlinearity: Literal["tanh", "relu", "deadzone", "sat"] = "tanh"


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

        self.la = torch.nn.Parameter(torch.zeros((self.nz,)))

        self.A_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))
        self.B = torch.nn.Parameter(
            torch.normal(0, 1 / self.nx, size=(self.nx, self.nd))
        )
        self.B2_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nw)))

        self.C = torch.nn.Parameter(
            torch.normal(0, 1 / self.ne, size=(self.ne, self.nx))
        )
        self.D = torch.nn.Parameter(
            torch.normal(0, 1 / self.ne, size=(self.ne, self.nd))
        )
        self.D12 = torch.nn.Parameter(
            torch.normal(0, 1 / self.ne, size=(self.ne, self.nw))
        )

        self.C2 = torch.nn.Parameter(torch.zeros((self.nz, self.nx)))
        self.D21 = torch.nn.Parameter(
            torch.normal(0, 1 / self.nz, size=(self.nz, self.nd))
        )
        self.D22 = torch.zeros((self.nz, self.nw))

        self.X = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))

    def set_lure_system(self) -> LureSystemClass:
        X_inv = torch.linalg.inv(self.X)
        A = X_inv @ self.A_tilde
        B2 = X_inv @ self.B2_tilde

        theta = trans.torch_bmat(
            [[A, self.B, B2], [self.C, self.D, self.D12], [self.C2, self.D21, self.D22]]
        )
        sys = get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = LureSystem(sys)

        return sys

    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    def forward(
        self, d: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        assert self.lure._nu == nu
        if x0 is None:
            x0 = torch.zeros(size=(B, self.nx))
        ds = d.reshape(shape=(B, N, nu, 1))
        es_hat, x = self.lure.forward(x0=x0, us=ds, return_states=True)
        return (es_hat.reshape(B, N, self.lure._ny), x.reshape(B, self.nx))

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


class Linear(nn.Module):
    def __init__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
    ) -> None:
        super().__init__()
        self._nx = A.shape[0]
        self._nu = B.shape[1]
        self._ny = C.shape[0]

        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def _init_weights(self) -> None:
        for p in self.parameters():
            torch.nn.init.uniform_(
                tensor=p, a=-np.sqrt(1 / self._nu), b=np.sqrt(1 / self._nu)
            )

    def state_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.A @ x + self.B @ u

    def output_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.C @ x + self.D @ u

    def forward(
        self, x0: torch.Tensor, us: torch.Tensor, return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, _, _ = us.shape
        x = torch.zeros(size=(n_batch, N + 1, self._nx, 1))
        y = torch.zeros(size=(n_batch, N, self._ny, 1))
        x[:, 0, :, :] = x0

        for k in range(N):
            x[:, k + 1, :, :] = self.state_dynamics(x=x[:, k, :, :], u=us[:, k, :, :])
            y[:, k, :, :] = self.output_dynamics(x=x[:, k, :, :], u=us[:, k, :, :])
        if return_state:
            return (y, x)
        else:
            return y


class LureSystem(Linear):
    def __init__(
        self,
        sys: LureSystemClass,
    ) -> None:
        super().__init__(A=sys.A, B=sys.B, C=sys.C, D=sys.D)
        self._nw = sys.B2.shape[1]
        self._nz = sys.C2.shape[0]
        assert self._nw == self._nz
        self.B2 = sys.B2
        self.C2 = sys.C2
        self.D12 = sys.D12
        self.D21 = sys.D21
        self.Delta = sys.Delta  # static nonlinearity

    def forward(
        self, x0: torch.Tensor, us: torch.Tensor, return_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, _, _ = us.shape
        y = torch.zeros(size=(n_batch, N, self._ny, 1))
        x = x0.reshape(n_batch, self._nx, 1)

        for k in range(N):
            w = self.Delta(self.C2 @ x + self.D21 @ us[:, k, :, :])
            x = super().state_dynamics(x=x, u=us[:, k, :, :]) + self.B2 @ w
            y[:, k, :, :] = (
                super().output_dynamics(x=x, u=us[:, k, :, :]) + self.D12 @ w
            )
        if return_states:
            return (y, x)
        else:
            return y


def load_model(model: ConstrainedModule, model_file_name: str) -> ConstrainedModule:
    model.load_state_dict(torch.load(model_file_name))
    model.set_lure_system()
    return model
