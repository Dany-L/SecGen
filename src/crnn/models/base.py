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

    def set_lure_system(self) -> LureSystemClass:
        X_inv = torch.linalg.inv(self.X)
        A = X_inv @ self.A_tilde
        B2 = X_inv @ self.B2_tilde

        C2 = torch.linalg.inv(self.get_L()) @ self.C2_tilde

        theta = trans.torch_bmat(
            [[A, self.B, B2], [self.C, self.D, self.D12], [C2, self.D21, self.D22]]
        )
        sys = get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = LureSystem(sys)

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

    def set_lure_system(self) -> LureSystemClass:

        D21 = torch.linalg.inv(self.get_L()) @ self.D21_tilde

        theta = trans.torch_bmat(
            [
                [self.A, self.B, self.B2],
                [self.C, self.D, self.D12],
                [self.C2, D21, self.D22],
            ]
        )
        sys = get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = LureSystem(sys)

        return sys


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
