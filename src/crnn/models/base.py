from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
from jax import Array
from jax.typing import ArrayLike
from nfoursid.nfoursid import NFourSID
from numpy.typing import NDArray
from pydantic import BaseModel
import cvxpy as cp
import scipy.linalg as la
import scipy.signal as sig
import pandas as pd

from ..configuration.base import InitializationData

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


class DynamicIdentificationModel(ABC):

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
        theta: Optional[ArrayLike] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        pass

    def check_constraints(self) -> bool:
        return True

    def get_phi(
        self, t: float, theta: Optional[ArrayLike] = None
    ) -> Union[torch.Tensor, Array]:
        return torch.tensor(0.0)

    @abstractmethod
    def project_parameters(self) -> str:
        pass

    def get_number_of_parameters(self) -> int:
        return -1


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
        theta: Optional[ArrayLike] = None,
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
        theta: Optional[ArrayLike] = None,
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
) -> Linear:

    N, nd = ds[0].shape
    M = len(ds)
    _, ne = es[0].shape

    f = MAX_SAMPLES_N4SID / (N * M)
    if f < 1:
        M_sub = np.max([int(f * M), 1])
        idx = np.random.choice(range(M), M_sub)
    else:
        idx = range(M)

    input_names = [f"u_{i}" for i in range(nd)]
    output_names = [f"y_{i}" for i in range(ne)]

    if N > MAX_SAMPLES_N4SID:
        N = MAX_SAMPLES_N4SID

    ds_sub, es_sub = [], []
    for i in idx:
        ds_sub.append(ds[i][:N])
        es_sub.append(es[i][:N])

    d = np.vstack(ds_sub)
    e = np.vstack(es_sub)

    if nx is None:
        nx = int(np.max([len(input_names), len(output_names)]))

    # io_data = pd.DataFrame(np.hstack([d, e]), columns=input_names + output_names)
    # n4sid = NFourSID(
    #     io_data,
    #     input_columns=input_names,
    #     output_columns=output_names,
    #     num_block_rows=num_block_rows
    # )

    A, B, C, D, Cov, S = N4SID(d.T, e.T, nx, require_stable=True)

    # n4sid.subspace_identification()
    # ss, _ = n4sid.system_identification(rank=nx)

    return Linear(
        torch.tensor(A), torch.tensor(B), torch.tensor(C), torch.tensor(D), dt
    )


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
        NumRows = 2 * nx
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
