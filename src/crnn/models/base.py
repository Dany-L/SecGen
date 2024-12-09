from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Type, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from pydantic import BaseModel

from ..utils import transformation as trans


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


class DynamicIdentificationModel(ABC):

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

    @abstractmethod
    def sdp_constraints(self) -> List[Callable]:
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
        pass

    def check_constraints(self) -> bool:
        return True

    def get_phi(self, t: float) -> torch.Tensor:
        return torch.tensor(0.0)

    @abstractmethod
    def project_parameters(self) -> str:
        pass

    def get_number_of_parameters(self) -> int:
        return -1


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


def load_model(
    model: DynamicIdentificationModel, model_file_name: str
) -> DynamicIdentificationModel:
    model.load_state_dict(
        torch.load(model_file_name, map_location=torch.device("cpu"), weights_only=True)
    )
    model.set_lure_system()
    return model


def retrieve_model_class(model_class_string: str) -> Type[DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split(".")
    module_string = ".".join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, DynamicIdentificationModel):
        raise ValueError(f"{cls} is not a subclass of DynamicIdentificationModel")
    return cls  # type: ignore
