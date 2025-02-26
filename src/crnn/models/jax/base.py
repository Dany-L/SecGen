from functools import partial
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple, Type,
                    Union)

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import torch
from jax import Array, jit, random
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .. import base as base


class ConstrainedModuleConfig(base.DynamicIdentificationConfig):
    sdp_opt: str = cp.MOSEK
    nonlinearity: Literal["tanh", "relu", "deadzone", "sat"] = "tanh"
    multiplier: Literal["none", "diag", "zf"] = "none"


class ConstrainedModule(base.DynamicIdentificationModel):
    def __init__(self, config: ConstrainedModuleConfig) -> None:
        super().__init__(config)

        self.nl = None
        if config.nonlinearity == "tanh":
            self.nl = jnp.tanh
        elif config.nonlinearity == "relu":
            self.nl = lambda x: jnp.maximum(0, x)
        else:
            raise ValueError(f"Unsupported nonlinearity: {config.nonlinearity}")

        self.sdp_opt = config.sdp_opt

        self.multiplier = config.multiplier
        if self.multiplier == "none":
            self.L = jnp.eye(self.nz)
        elif config.multiplier == "diag":
            self.L = jnp.ones((self.nz,))
        else:
            raise NotImplementedError(f"Unsupported multiplier: {config.multiplier}")

        self.parameter_names = ["A", "B", "B2", "C", "D", "D12", "C2", "D21", "D22"]
        self.parameter_sizes = [
            (self.nx, self.nx),
            (self.nx, self.nd),
            (self.nx, self.nw),
            (self.ne, self.nx),
            (self.ne, self.nd),
            (self.ne, self.nw),
            (self.nz, self.nx),
            (self.nz, self.nd),
            (self.nz, self.nw),
        ]
        key = random.key(0)

        self.theta = random.normal(key, (self.get_number_of_parameters(), 1))
        self.theta = self.initialize_parameters()

    def initialize_parameters(
        self,
        ds: List[NDArray[np.float64]],
        es: List[NDArray[np.float64]],
        init_data: Dict[str, Any],
    ) -> base.InitializationData:
        self.theta = jnp.zeros((self.get_number_of_parameters(), 1))
        return base.InitializationData("initialized theta with zeros", {})

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

    def get_L(self, L: ArrayLike) -> Array:
        if self.multiplier == "none":
            return L
        elif self.multiplier == "diag":
            return jnp.diag(L[:, 0])
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")

    def set_L(self, L: ArrayLike) -> Array:
        if self.multiplier == "none":
            L = L
        elif self.multiplier == "diag":
            L = jnp.diag(L)
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")
        return L

    def check_constraints(self, theta: Optional[ArrayLike] = None) -> bool:
        # check if constraints are psd
        if theta is None:
            theta = self.theta
        for lmi in self.sdp_constraints(theta):
            try:
                jnp.linalg.cholesky(lmi())
            except np.linalg.LinAlgError:
                return False

        for scalar in self.pointwise_constraints():
            if scalar() < 0:
                return False
        return True

    def sdp_constraints(self) -> List[Callable]:
        return [lambda: np.eye(self.nz)]

    def pointwise_constraints(self) -> List[Callable]:
        return [lambda: 1.0]

    def get_phi(
        self, t: float, theta: Optional[ArrayLike] = None
    ) -> Union[torch.Tensor, Array]:
        if theta is None:
            theta = self.theta
        batch_phi = 0.0
        for M in self.sdp_constraints(theta):
            _, slogdet = jnp.linalg.slogdet(M())
            batch_phi += -1 / t * slogdet

        return batch_phi

    def get_number_of_parameters(self):
        return sum([np.prod(p) for p in self.parameter_sizes])


def load_model(model: ConstrainedModule, model_file_name: str) -> ConstrainedModule:

    params_dict = jnp.load(model_file_name)
    theta_list = [
        jnp.array(params_dict[n]).flatten().reshape(-1, 1)
        for n in model.parameter_names
    ]
    model.theta = jnp.vstack(theta_list)
    return model


def retrieve_model_class(
    model_class_string: str,
) -> Type[base.DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split(".")
    module_string = ".".join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, base.DynamicIdentificationModel):
        raise ValueError(f"{cls} is not a subclass of DynamicIdentificationModel")
    return cls  # type: ignore


def init_layer_params(nx, nd, nw, n_out, key, scale=1e-2) -> Array:
    x_key, d_key, w_key = random.split(key, 3)
    # return (
    #     scale * random.normal(x_key, (n_out, nx)),
    #     scale * random.normal(d_key, (n_out, nd)),
    #     scale * random.normal(w_key, (n_out, nw)),
    # )
    return (jnp.zeros((n_out, nx)), jnp.zeros((n_out, nd)), jnp.zeros((n_out, nw)))


def init_network_params(
    n_outs: List[int], key: Array, nx: int, nd: int, nw: int
) -> List[Array]:
    pars = []
    keys = random.split(key, len(n_outs))
    for n_out, key in zip(n_outs, keys):
        pars.extend(init_layer_params(nx, nd, nw, n_out, key))

    return pars


@partial(jit, static_argnames=["dF"])
def update(
    theta: ArrayLike,
    dF: Callable[[ArrayLike], Array],
    s=0.01,
) -> List[Array]:
    if isinstance(theta, list):
        return [p - s * dP for p, dP in zip(theta, dF(theta))]
    else:
        return theta - s * dF(theta)
