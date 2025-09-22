from abc import abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cvxpy as cp

from nfoursid.nfoursid import NFourSID
from numpy.typing import NDArray
from pydantic import BaseModel
from typing import Literal

from ..configuration.base import InitializationData
from ..utils import transformation as trans
from ..utils import base as utils

MAX_SAMPLES_N4SID = 45000


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


class DynamicIdentificationModel(nn.Module):

    def __init__(self, config: DynamicIdentificationConfig) -> None:
        super().__init__()
        self.nz, self.nw = config.nz, config.nz  # size of nonlinear channel
        # internal state size matches size of nonlinear channel for this network
        self.nx = self.nz
        self.nd, self.ne = config.nd, config.ne  # size of input and output

    def initialize_parameters(
        self,
        ds: List[NDArray[np.float64]],
        es: List[NDArray[np.float64]],
        init_data: Dict[str, Any],
    ) -> InitializationData:
        self.set_lure_system()
        return InitializationData("Standard initialization of parameters.", {})

    @abstractmethod
    def sdp_constraints(self) -> List[Callable]:
        pass

    @abstractmethod
    def pointwise_constraints(self) -> List[Callable]:
        pass

    def set_lure_system(self) -> LureSystemClass:
        lure_matrices = get_lure_matrices(
            torch.zeros(
                size=(self.nx + self.ne + self.nw, self.nx + self.nd + self.nz)
            ),
            self.nx,
            self.nd,
            self.ne,
        )
        self.lure = LureSystem(lure_matrices)
        return lure_matrices

    @abstractmethod
    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[np.ndarray] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        pass

    def check_constraints(self) -> bool:
        return True

    def get_phi(
        self, t: float, theta: Optional[np.ndarray] = None
    ) -> Union[torch.Tensor, torch.Tensor]:
        return torch.tensor(0.0)

    @abstractmethod
    def project_parameters(self) -> str:
        pass

    def get_number_of_parameters(self) -> int:
        return -1

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def pointwise_constraints(self) -> List[Callable]:
        return [lambda: torch.tensor(1)]

    def sdp_constraints(self) -> List[Callable]:
        return [lambda: torch.eye(self.nz)]


class Linear(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt: torch.Tensor = torch.tensor(0.01),
    ) -> None:
        super().__init__()
        self._nx = A.shape[0]
        self._nu = B.shape[1]
        self._ny = C.shape[0]

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt

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
        self,
        d: torch.Tensor,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[np.ndarray] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        n_batch, N, _, _ = d.shape
        x = torch.zeros(size=(n_batch, N + 1, self._nx, 1))
        y = torch.zeros(size=(n_batch, N, self._ny, 1))
        if x0 is not None:
            if isinstance(x0, tuple):
                x[:, 0, :, :] = x0[0]  # Use first element if tuple
            else:
                x[:, 0, :, :] = x0

        for k in range(N):
            x[:, k + 1, :, :] = self.state_dynamics(x=x[:, k, :, :], u=d[:, k, :, :])
            y[:, k, :, :] = self.output_dynamics(x=x[:, k, :, :], u=d[:, k, :, :])

        return y, (x,)

    def is_stable(self) -> bool:
        """Check if the system is stable by checking the eigenvalues of A."""
        eigenvalues = torch.linalg.eigvals(self.A)
        return bool(torch.all(torch.abs(eigenvalues) < 1.0))


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
        self,
        d: torch.Tensor,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[np.ndarray] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        n_batch, N, _, _ = d.shape
        y = torch.zeros(size=(n_batch, N, self._ny, 1))
        x = x0.reshape(n_batch, self._nx, 1)

        for k in range(N):
            w = self.Delta(self.C2 @ x + self.D21 @ d[:, k, :, :])
            x = super().state_dynamics(x=x, u=d[:, k, :, :]) + self.B2 @ w
            y[:, k, :, :] = super().output_dynamics(x=x, u=d[:, k, :, :]) + self.D12 @ w

        return (y, x)


class ConstrainedModuleConfig(DynamicIdentificationConfig):
    sdp_opt: str = cp.MOSEK
    nonlinearity: Literal["tanh", "relu", "deadzone", "sat"] = "tanh"
    multiplier: Literal["none", "diag", "zf"] = "none"
    ga2: float = 1.0
    init_type: Literal["rand", "zero"] = "rand"
    init_std: float = 1.0
    learn_H: bool = False
    initialization: Literal["n4sid", "project"] = "project"
    nx: Optional[int] = None


class ConstrainedModule(DynamicIdentificationModel):
    def __init__(self, config: ConstrainedModuleConfig) -> None:
        super().__init__(config)

        if config.nx:
            self.nx = config.nx

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
            self.L = torch.nn.Parameter(config.init_std * torch.ones((self.nz,)))
        else:
            raise NotImplementedError(f"Unsupported multiplier: {config.multiplier}")

    def get_optimization_multiplier_and_constraints(
        self,
    ) -> Tuple[Union[NDArray[np.float64], cp.Variable], List[cp.Constraint]]:
        if self.multiplier == "none":
            L = np.eye(self.nz)
            multiplier_constraints = []
        elif self.multiplier == "diag":
            lam = cp.Variable((self.nz,))
            L = cp.diag(lam)
            multiplier_constraints = [lam_i >= 0 for lam_i in lam]
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")
        return (L, multiplier_constraints)

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[np.ndarray] = None,
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
        es_hat, x = self.lure.forward(x0=x0, d=ds)
        return (
            es_hat.reshape(B, N, self.lure._ny),
            (x.reshape(B, self.nx),),
        )

    def check_constraints(self) -> bool:
        # check if constraints are psd
        with torch.no_grad():
            for lmi in self.sdp_constraints():
                # min_ev = torch.linalg.eigh(lmi()).eigenvalues[0]
                # if min_ev < 0:
                #     return False
                _, info = torch.linalg.cholesky_ex(lmi())

                if info > 0:
                    return False

            for scalar in self.pointwise_constraints():
                if scalar() < 0:
                    return False
        return True

    def get_phi(
        self, t: float, theta: Optional[np.ndarray] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        phi = torch.tensor(0.0)
        for F_i in self.sdp_constraints():
            phi += -torch.logdet(F_i())
        for F_i in self.pointwise_constraints():
            phi += -torch.log(F_i())
        return 1 / t * phi


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


class L2StableConstrainedModule(ConstrainedModule):
    def __init__(self, config: ConstrainedModuleConfig) -> None:
        super().__init__(config)
        self.A_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))
        self.B_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nd)))
        self.B2_tilde = torch.nn.Parameter(torch.zeros((self.nx, self.nw)))

        self.C = torch.nn.Parameter(torch.zeros((self.ne, self.nx)))
        self.D = torch.nn.Parameter(torch.zeros((self.ne, self.nd)))
        self.D12 = torch.nn.Parameter(torch.zeros((self.ne, self.nw)))

        self.C2_tilde = torch.nn.Parameter(torch.zeros((self.nz, self.nx)))
        self.D21_tilde = torch.nn.Parameter(torch.zeros((self.nz, self.nd)))
        self.D22 = torch.zeros((self.nz, self.nw))

        self.initialization = config.initialization

        # X is required to be symmetric. We use a lower triangular matrix as parameter.
        # X = L @ L.T then ensures symmetry.
        self.Lx = torch.nn.Parameter(config.init_std * torch.eye(self.nx, self.nx))

        # self.ga2 = torch.nn.Parameter(torch.tensor([[config.ga2]]), requires_grad=False)
        self.ga2 = torch.tensor([[config.ga2]])

        if config.init_type == "rand":
            for n, p in self.named_parameters():
                if p.requires_grad and not (n == "X" or n == "L"):
                    torch.nn.init.normal_(p, mean=0.0, std=config.init_std)

    def initialize_parameters(
        self,
        ds: List[NDArray[np.float64]],
        es: List[NDArray[np.float64]],
        init_data: Dict[str, Any],
    ) -> InitializationData:

        if self.initialization == "project":
            msg = self.project_parameters()
            self.set_lure_system()
            return InitializationData(msg, {})
        elif self.initialization == "n4sid":
            if init_data:
                msg = "n4sid: Initialization loaded from file"
                ss = init_data["ss"]
            else:
                start_time = time.time()
                ss = run_n4sid(ds, es, self.nx, None)
                stop_time = time.time()
                n4sid_duration = utils.get_duration_str(start_time, stop_time)
                msg = f"n4sid: duration {n4sid_duration}"

            from ..systemtheory.base import AnalysisLti, SectorBounded

            an = AnalysisLti(ss, self.nz)
            msg_l2, add_par = an.l2(
                SectorBounded(0, 1), self.ga2.cpu().detach().numpy()
            )
            msg += f" l2 analysis: {msg_l2}"
            assert add_par.ga2 <= self.ga2.cpu().detach().numpy()

            self.Lx.data = torch.linalg.cholesky(torch.tensor(add_par.X))
            assert (
                np.linalg.norm((self.Lx @ self.Lx.T).cpu().detach().numpy() - add_par.X)
            ) < 1e-5

            self.set_L(torch.tensor(add_par.Lambda))

            self.A_tilde.data = self.get_X() @ ss.A
            self.B_tilde.data = self.get_X() @ ss.B
            self.C.data = ss.C
            self.D.data = ss.D

            # self.A_tilde.requires_grad = False
            # self.B_tilde.requires_grad = False
            # self.C.requires_grad = False
            # self.D.requires_grad = False

            self.set_lure_system()
            msg_proj = self.project_parameters()
            msg += f" proj: {msg_proj}"
            self.set_lure_system()
            # assert self.check_constraints()

            return InitializationData(msg, {"ss": ss})
        else:
            raise NotImplementedError(
                f"Initialization type {self.initialization} is not implemented."
            )

    def set_lure_system(self) -> LureSystemClass:
        X_inv = torch.linalg.inv(self.get_X())
        A = X_inv @ self.A_tilde
        B = X_inv @ self.B_tilde
        B2 = X_inv @ self.B2_tilde

        L_inv = torch.linalg.inv(self.get_L())
        C2 = L_inv @ self.C2_tilde
        D21 = L_inv @ self.D21_tilde

        theta = trans.torch_bmat(
            [[A, B, B2], [self.C, self.D, self.D12], [C2, D21, self.D22]]
        )
        sys = get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = LureSystem(sys)

        return sys

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
            self.L.data = torch.diag(L)
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")

    def get_X(self) -> torch.Tensor:
        return self.Lx @ self.Lx.T


class OptFcn:
    def __init__(
        self,
        d: torch.Tensor,
        e: torch.Tensor,
        nn: ConstrainedModule,
        t=float,
        x0: Optional[torch.Tensor] = None,
        loss: torch.nn.Module = torch.nn.MSELoss(),
    ) -> None:
        self.d = d
        self.x0 = x0
        self.e = e
        self.nn = nn
        self.loss = loss
        self.t = torch.tensor(t)

    def f(self, theta: torch.Tensor) -> torch.Tensor:
        self.set_vec_pars_to_model(theta)
        # print(torch.hstack([p.squeeze() for p in self.nn.parameters()]))
        self.nn.set_lure_system()
        loss = self.loss(self.nn(self.d, self.x0)[0], self.e)
        phi = self.nn.get_phi(self.t)
        return loss + phi

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


def get_lure_matrices(
    gen_plant: torch.Tensor,
    nx: int,  # state
    nd: int,  # input
    ne: int,  # output
    nonlinearity: Optional[torch.nn.Module] = None,
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

    if nonlinearity is None:
        nonlinearity = torch.nn.Tanh()

    return LureSystemClass(A, B, B2, C, D, D12, C2, D21, D22, nonlinearity)


def run_n4sid(
    ds: List[NDArray[np.float64]],
    es: List[NDArray[np.float64]],
    dt: np.float64,
    nx=None,
    N_max=None,
) -> Tuple[Linear, NDArray[np.float64], NDArray[np.float64]]:

    N, nd = ds[0].shape
    if N_max is not None:
        N = min(N, N_max)
    _, ne = es[0].shape

    input_names = [f"u_{i}" for i in range(nd)]
    output_names = [f"y_{i}" for i in range(ne)]

    d = np.vstack(ds)
    e = np.vstack(es)

    if nx is None:
        nx = int(np.max([len(input_names), len(output_names)]))

    A, B, C, D, Cov, S = N4SID_NG_with_nfoursid(
        d.T[:, :N], e.T[:, :N], nx, enforce_stability_method="projection"
    )

    # Evaluate the linear approximation
    e_hat = simulate_linear_system(A, B, C, D, d.T)

    # for i in range(ne):
    #     fig, ax = plt.subplots(figsize=(10,4))
    #     t = np.linspace(0,(N-1)*dt, N)
    #     ax.plot(t, e.T[i, :N], label="e")
    #     ax.plot(t, e_hat[i, :N], label="e_hat")
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Output")
    #     ax.legend()
    #     ax.grid()

    # Compute fit percentage
    fit_percent = compute_fit_percent(e.T, e_hat)

    # Compute RMSE
    rmse = np.sqrt(np.mean((e.T - e_hat) ** 2, axis=1))

    return (
        Linear(
            torch.tensor(A),
            torch.tensor(B),
            torch.tensor(C),
            torch.tensor(D),
            torch.tensor(dt),
        ),
        fit_percent,
        rmse,
    )


def N4SID_NG_with_nfoursid(
    u: np.ndarray,
    y: np.ndarray,
    nx: int,
    NumRows: Optional[int] = None,
    NumCols: Optional[int] = None,
    enforce_stability_method: str = "projection",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate discrete-time state-space model using the NFourSID package.

    Inputs:
        u, y           - Input/output data arrays (shape: [n_channels, time_steps])
        nx             - Desired model order (number of states)
        NumRows        - Number of block rows (past horizon)
        NumCols        - Number of block columns (future horizon)
        require_stable - If True, enforce discrete-time A-matrix stability
        enforce_stability_method - Stability enforcement method (not used here)

    Returns:
        A, B, C, D - State-space matrices
        Cov       - Covariance matrix of residuals
        Sigma     - Singular values from the SVD (for model order selection)
    """

    # Check that there's enough data
    NumVals = u.shape[1]

    if NumRows is None and NumCols is None:
        NumRows = 15 * nx
        NumCols = max(NumVals - 2 * NumRows + 1, 0)

    assert (
        NumVals >= 2 * NumRows + NumCols - 1
    ), "Insufficient data length for the given NumRows and NumCols."

    # Convert input/output data to a dataframe
    input_names = [f"u_{i}" for i in range(u.shape[0])]
    output_names = [f"y_{i}" for i in range(y.shape[0])]
    data = np.hstack([u.T, y.T])
    columns = input_names + output_names
    state_space_df = pd.DataFrame(data, columns=columns)

    # Initialize the NFourSID object
    nfoursid = NFourSID(
        state_space_df,
        output_columns=output_names,
        input_columns=input_names,
        num_block_rows=NumRows,
    )

    # Perform subspace identification
    nfoursid.subspace_identification()

    # Perform system identification with the specified model order
    state_space_identified, covariance_matrix = nfoursid.system_identification(rank=nx)

    if enforce_stability_method == "projection":
        # Fast stability enforcement: scale A to unit disc
        A = state_space_identified.a
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 1.0:
            A = A / (max_eig + 1e-6)
    else:
        raise ValueError(
            f"Unknown stability enforcement method: {enforce_stability_method}"
        )

    # Extract state-space matrices
    B = state_space_identified.b
    C = state_space_identified.c
    D = state_space_identified.d

    return A, B, C, D, covariance_matrix, None


def simulate_linear_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    u: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate output of a discrete-time linear system:
    x_{t+1} = A x_t + B u_t
    y_t = C x_t + D u_t
    u: shape [n_inputs, T]
    Returns y: shape [n_outputs, T]
    """
    nx = A.shape[0]
    n_outputs = C.shape[0]
    T = u.shape[1]
    x = np.zeros((nx, T))
    y = np.zeros((n_outputs, T))
    if x0 is None:
        x_prev = np.zeros((nx, 1))
    else:
        x_prev = x0.reshape(nx, 1)
    for t in range(T):
        u_t = u[:, t : t + 1]
        x[:, t : t + 1] = A @ x_prev + B @ u_t
        y[:, t : t + 1] = C @ x[:, t : t + 1] + D @ u_t
        x_prev = x[:, t : t + 1]
    return y


def compute_fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the fit percentage between true and predicted values.

    y_true : True output data
    y_pred : Predicted output data

    Returns:
        fit_percent : Fit percentage
    """
    fit_percent = 100 * (1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true**2))
    return fit_percent
