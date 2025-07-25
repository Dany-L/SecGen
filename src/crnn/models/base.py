from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import cvxpy as cp
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt

from nfoursid.nfoursid import NFourSID
from numpy.typing import NDArray
from pydantic import BaseModel
from typing import Literal

from ..configuration.base import InitializationData
from ..utils import transformation as trans

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
        dt: torch.tensor = 0.0,
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
        x[:, 0, :, :] = x0

        for k in range(N):
            x[:, k + 1, :, :] = self.state_dynamics(x=x[:, k, :, :], u=d[:, k, :, :])
            y[:, k, :, :] = self.output_dynamics(x=x[:, k, :, :], u=d[:, k, :, :])

        return (y, x)


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
    learn_H: bool = (False,)
    initialization: Literal["n4sid", "project"] = "project"
    nx: Optional[int] = False


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
            la = cp.Variable((self.nz,))
            L = cp.diag(la)
            multiplier_constraints = [lam >= 0 for lam in la]
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

            an = ana.AnalysisLti(ss, self.nz)
            msg_l2, add_par = an.l2(
                ana.SectorBounded(0, 1), self.ga2.cpu().detach().numpy()
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
        x0: torch.Tensor = None,
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
    nonlinearity: torch.nn.Module = torch.nn.Tanh(),
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


def run_n4sid(
    ds: NDArray[np.float64],
    es: NDArray[np.float64],
    dt: np.float64,
    nx=None,
    num_block_rows=1,
    N_max = None
) -> Tuple[Linear, NDArray[np.float64], NDArray[np.float64]]:

    N, nd = ds[0].shape
    if N_max is not None:
        N = min(N, N_max)
    M = len(ds)
    _, ne = es[0].shape

    input_names = [f"u_{i}" for i in range(nd)]
    output_names = [f"y_{i}" for i in range(ne)]

    d = np.vstack(ds)
    e = np.vstack(es)

    if nx is None:
        nx = int(np.max([len(input_names), len(output_names)]))

    A, B, C, D, Cov, S = N4SID_NG_with_nfoursid(d.T[:, :N], e.T[:,:N], nx, enforce_stability_method='projection')

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

    return (Linear(
        torch.tensor(A), torch.tensor(B), torch.tensor(C), torch.tensor(D), dt
    ), fit_percent, rmse)


def N4SID(
    u: np.ndarray,
    y: np.ndarray,
    nx: int,
    NumRows: int = None,
    NumCols: int = None,
    require_stable: bool = False,
    enforce_stability_method: str = "cvxpy"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate discrete-time state-space model using the N4SID method.

    Inputs:
        u, y           - Input/output data arrays (shape: [n_channels, time_steps])
        nx             - Desired model order (number of states)
        NumRows        - Number of block rows (past horizon)
        NumCols        - Number of block columns (future horizon)
        require_stable - If True, enforce discrete-time A-matrix stability
        enforce_stability_method - "cvxpy" (slow, constrained) or "projection" (fast)

    Returns:
        A, B, C, D - State-space matrices
        Cov       - Covariance matrix of residuals
        Sigma     - Singular values from the SVD (for model order selection)
    """

    NumInputs = u.shape[0]
    NumOutputs = y.shape[0]

    # Check that there's enough data
    NumVals = u.shape[1]

    if NumRows is None and NumCols is None:
        NumRows = 15 * nx
        NumCols = NumVals - 2 * NumRows + 1

    assert (
        NumVals >= 2 * NumRows + NumCols - 1
    ), "Insufficient data length for the given NumRows and NumCols."

    # Build dimension dictionary for preprocessing
    NumDict = {
        "Inputs": NumInputs,
        "Outputs": NumOutputs,
        "Dimension": nx,
        "Rows": NumRows,
        "Columns": NumCols,
    }

    # Preprocess the input/output to get regression matrices
    GammaDict, Sigma = preProcess(u, y, NumDict)

    GamData = GammaDict["Data"]  # Regressor matrix
    GamYData = GammaDict["DataY"]  # Target (shifted Gamma * state + Yfuture)

    if not require_stable:
        # Standard least squares regression
        K = la.lstsq(GamData.T, GamYData.T)[0].T
    elif enforce_stability_method == "cvxpy":
        # Constrained least squares with Lyapunov stability (A-matrix)
        Kvar = cp.Variable((nx + NumOutputs, GamData.shape[0]))
        Avar = Kvar[:nx, :nx]
        Pvar = cp.Variable((nx, nx), PSD=True)
        LyapBlock = cp.bmat([[Pvar, Avar], [Avar.T, np.eye(nx)]])
        constraints = [LyapBlock >> 0, Pvar << np.eye(nx)]
        residual = GamYData - Kvar @ GamData
        objective = cp.Minimize(cp.norm(residual, "fro"))
        problem = cp.Problem(objective, constraints)
        result = problem.solve()
        if Kvar.value is None:
            raise RuntimeError("CVXPY failed to solve the stability-constrained problem.")
        K = Kvar.value
    elif enforce_stability_method == "projection":
        # Fast stability enforcement: scale A to unit disc
        K = la.lstsq(GamData.T, GamYData.T)[0].T
        A = K[:nx, :nx]
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 1.0:
            K[:nx, :nx] = A / (max_eig + 1e-6)
    else:
        raise ValueError(f"Unknown stability enforcement method: {enforce_stability_method}")

    A, B, C, D, Cov = postProcess(K, GammaDict, NumDict)
    return A, B, C, D, Cov, Sigma


def getHankelMatrices(x, NumRows, NumCols, blockWidth=1):
    """
    Creates past and future block Hankel matrices from signal `x`.
    x: signal array of shape [n_signals, T]
    Returns: XPast, XFuture
    """
    bh = x.shape[0]
    xPastLeft = blockTranspose(x[:, :NumRows], blockHeight=bh, blockWidth=blockWidth)
    XPast = blockHankel(
        xPastLeft, x[:, NumRows - 1 : NumRows - 1 + NumCols], blockHeight=bh
    )

    xFutureLeft = blockTranspose(
        x[:, NumRows : 2 * NumRows], blockHeight=bh, blockWidth=blockWidth
    )
    XFuture = blockHankel(
        xFutureLeft, x[:, 2 * NumRows - 1 : 2 * NumRows - 1 + NumCols], blockHeight=bh
    )

    return XPast, XFuture


def preProcess(u, y, NumDict):
    NumInputs = u.shape[0]
    NumOutputs = y.shape[0]
    NumRows = NumDict["Rows"]
    NumCols = NumDict["Columns"]
    NSig = NumDict["Dimension"]

    UPast, UFuture = getHankelMatrices(u, NumRows, NumCols) # U_0|i-1, U_i|2i-1
    YPast, YFuture = getHankelMatrices(y, NumRows, NumCols) # Y_0|i-1

    Data = np.vstack((UPast, UFuture, YPast)) # Z_i
    # Use lstsq with rcond=None for best performance
    L, _, _, _ = la.lstsq(Data.T, YFuture.T, lapack_driver='gelsy')
    L = L.T
    Z = L @ Data

    DataShift = np.vstack((UPast, UFuture[NumInputs:], YPast))
    LShift, _, _, _ = la.lstsq(DataShift.T, YFuture[NumOutputs:].T, lapack_driver='gelsy')
    LShift = LShift.T
    ZShift = LShift @ DataShift

    L1 = L[:, : NumInputs * NumRows]
    L3 = L[:, 2 * NumInputs * NumRows :]

    LPast = np.hstack((L1, L3))
    DataPast = np.vstack((UPast, YPast))

    # Use economy SVD
    U_svd, S, Vt = la.svd(LPast @ DataPast, full_matrices=False)
    Gamma = U_svd[:, :NSig] * np.sqrt(S[:NSig])
    GammaLess = Gamma[:-NumOutputs, :]

    GammaPinv = la.pinv(Gamma, atol=1e-8)
    GammaLessPinv = la.pinv(GammaLess, atol=1e-8)

    GamShiftSolve, _, _, _ = la.lstsq(GammaLess, ZShift, lapack_driver='gelsy')
    GamSolve, _, _, _ = la.lstsq(Gamma, Z, lapack_driver='gelsy')

    GamData = np.vstack((GamSolve, UFuture))
    GamYData = np.vstack((GamShiftSolve, YFuture[:NumOutputs]))

    GammaDict = {
        "Data": GamData,
        "DataLess": GammaLess,
        "DataY": GamYData,
        "Pinv": GammaPinv,
        "LessPinv": GammaLessPinv,
    }
    return GammaDict, S


def blockTranspose(M, blockHeight, blockWidth):
    """
    Rearranges blocks in a matrix (no transpose of inner blocks).
    Converts (r x c) into (blockHeight*blockCols x blockWidth*blockRows).
    """
    r, c = M.shape
    assert r % blockHeight == 0 and c % blockWidth == 0, "Incompatible block size"
    Nr = r // blockHeight
    Nc = c // blockWidth

    Mblock = np.zeros((Nr, Nc, blockHeight, blockWidth))
    for i in range(Nr):
        for j in range(Nc):
            Mblock[i, j] = M[
                i * blockHeight : (i + 1) * blockHeight,
                j * blockWidth : (j + 1) * blockWidth,
            ]

    MtBlock = np.transpose(Mblock, (1, 0, 2, 3))  # Swap blocks (rows <-> cols)
    return block2mat(MtBlock)


def blockHankel(Hleft, Hbot=None, blockHeight=1):
    """
    Constructs a block Hankel matrix from left and optional bottom blocks.
    Hleft: (Nr * bh) x bw
    Hbot: optional (bh x Nc * bw)
    Returns: full Hankel matrix as 2D array
    """
    blockWidth = Hleft.shape[1]
    Nr = Hleft.shape[0] // blockHeight
    Nc = Nr if Hbot is None else Hbot.shape[1] // blockWidth

    # Extract blocks efficiently
    LeftBlock = Hleft.reshape(Nr, blockHeight, blockWidth)
    MBlock = np.zeros((Nr, Nc, blockHeight, blockWidth), dtype=Hleft.dtype)

    for k in range(min(Nc, Nr)):
        MBlock[: Nr - k, k] = LeftBlock[k:]

    if Hbot is not None:
        BotBlock = np.zeros((Nc, blockHeight, blockWidth))
        for i in range(Nc):
            BotBlock[i] = Hbot[:, i * blockWidth : (i + 1) * blockWidth]

        for k in range(max(1, Nc - Nr), Nc):
            MBlock[Nr - Nc + k, Nc - k :] = BotBlock[1 : k + 1]

    return block2mat(MBlock)


def block2mat(Mblock):
    # Efficiently reshape the block array to 2D matrix
    Nr, Nc, bh, bw = Mblock.shape
    return Mblock.transpose(0, 2, 1, 3).reshape(Nr * bh, Nc * bw)


def postProcess(K, GammaDict, NumDict):
    """
    Recover state-space matrices (A, B, C, D) and output noise covariance from identified data.

    Parameters:
        K          : System matrix from preProcess or optimization [ (n + p) x (n + mN) ]
        GammaDict  : Dictionary with state sequence and its pseudoinverses from preProcess
        NumDict    : Dictionary with model dimensions: Rows, Columns, Inputs, Outputs, Dimension

    Returns:
        AID, BID, CID, DID, CovID
    """
    GamData = GammaDict["Data"]
    GamYData = GammaDict["DataY"]
    GammaPinv = GammaDict["Pinv"]
    GammaLessPinv = GammaDict["LessPinv"]
    GammaLess = GammaDict["DataLess"]

    NSig = NumDict["Dimension"]
    NumRows = NumDict["Rows"]
    NumCols = NumDict["Columns"]
    NumInputs = NumDict["Inputs"]
    NumOutputs = NumDict["Outputs"]

    # --- Extract A and C
    AID = K[:NSig, :NSig]
    CID = K[NSig:, :NSig]

    # --- Innovation covariance
    rho = GamYData - K @ GamData
    CovID = np.dot(rho, rho.T) / NumCols

    # --- Build L matrix
    AC = np.vstack((AID, CID))
    L = AC @ GammaPinv

    # --- Build Hankel subtraction matrices (vectorized)
    M = np.zeros((NSig, NumRows * NumOutputs))
    M[:, NumOutputs:] = GammaLessPinv

    Mleft = blockTranspose(M, NSig, NumOutputs)
    LtopLeft = blockTranspose(L[:NSig], NSig, NumOutputs)

    NTop = blockHankel(Mleft, blockHeight=NSig) - blockHankel(LtopLeft, blockHeight=NSig)

    LbotLeft = blockTranspose(L[NSig:], NumOutputs, NumOutputs)
    NBot = -blockHankel(LbotLeft, blockHeight=NumOutputs)
    NBot[:NumOutputs, :NumOutputs] += np.eye(NumOutputs)

    N = np.vstack((NTop, NBot)) @ la.block_diag(np.eye(NumOutputs), GammaLess)

    # --- Build input gain matrix from K (vectorized)
    Kr = K[:, NSig:]
    KsTop = Kr[:NSig].reshape(NSig, NumRows, NumInputs).transpose(1, 0, 2).reshape(NSig * NumRows, NumInputs)
    KsBot = Kr[NSig:].reshape(NumOutputs, NumRows, NumInputs).transpose(1, 0, 2).reshape(NumOutputs * NumRows, NumInputs)
    Ks = np.vstack((KsTop, KsBot))

    DB, _, _, _ = la.lstsq(N, Ks, lapack_driver='gelsy')
    DID = DB[:NumOutputs]
    BID = DB[NumOutputs:]

    return AID, BID, CID, DID, CovID


def N4SID_NG_with_nfoursid(
    u: np.ndarray,
    y: np.ndarray,
    nx: int,
    NumRows: int = None,
    NumCols: int = None,
    enforce_stability_method: str = "projection"
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
        NumCols = NumVals - 2 * NumRows + 1

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
        num_block_rows=NumRows
    )

    # Perform subspace identification
    nfoursid.subspace_identification()

    # Perform system identification with the specified model order
    state_space_identified, covariance_matrix = nfoursid.system_identification(
        rank=nx
    )

    if enforce_stability_method == "projection":
        # Fast stability enforcement: scale A to unit disc
        A = state_space_identified.a
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 1.0:
            A = A / (max_eig + 1e-6)
    else:
        raise ValueError(f"Unknown stability enforcement method: {enforce_stability_method}")

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
    x0: np.ndarray = None
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
        u_t = u[:, t:t+1]
        x[:, t:t+1] = A @ x_prev + B @ u_t
        y[:, t:t+1] = C @ x[:, t:t+1] + D @ u_t
        x_prev = x[:, t:t+1]
    return y


def compute_fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the fit percentage between true and predicted values.

    y_true : True output data
    y_pred : Predicted output data

    Returns:
        fit_percent : Fit percentage
    """
    fit_percent = 100 * (1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2))
    return fit_percent
