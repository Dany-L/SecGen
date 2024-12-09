from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Type, Union

import cvxpy as cp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from numpy.typing import NDArray
from pydantic import BaseModel

from ..utils import transformation as trans
from . import base as base


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

        self.parameter_names = [
            ["A", "B", "B2"],
            ["C", "D", "D12"],
            ["C2", "D21", "D22"],
        ]
        self.theta = jnp.zeros(
            (self.nx + self.ne + self.nz, self.nx + self.nd + self.nw)
        )

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

    def get_L(self) -> Array:
        if self.multiplier == "none":
            return self.L
        elif self.multiplier == "diag":
            return jnp.diag(self.L)
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")

    def set_L(self, L: ArrayLike) -> None:
        if self.multiplier == "none":
            self.L = L
        elif self.multiplier == "diag":
            self.L.value = jnp.diag(L)
        else:
            raise NotImplementedError(f"Unsupported multiplier: {self.multiplier}")

    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    def check_constraints(self) -> bool:
        # check if constraints are psd
        for lmi in self.sdp_constraints():
            _, info = jnp.linalg.cholesky(lmi())
            if info > 0:
                return False

        for scalar in self.pointwise_constraints():
            if scalar() < 0:
                return False
        return True

    def get_phi(self, t: float) -> Array:
        if self.sdp_constraints() is not None:
            batch_phi = (
                1
                / t
                * jnp.sum(jnp.array([-jnp.logdet(M()) for M in self.sdp_constraints()]))
            )
        else:
            batch_phi = jnp.array(0.0)
        return batch_phi

    def get_number_of_parameters(self):
        num_params = 0
        for par_rows in self.theta:
            num_params += sum([p.size for p in par_rows])
        return num_params


def load_model(model: ConstrainedModule, model_file_name: str) -> ConstrainedModule:

    params_dict = jnp.load(model_file_name)
    theta = []
    for names in model.parameter_names:
        theta.append([jnp.array(params_dict[n]) for n in names])
    model.theta = theta
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


def init_layer_params(nx, nd, nw, n_out, key) -> Array:
    x_key, d_key, w_key = random.split(key, 3)
    return (
        random.normal(x_key, (n_out, nx)),
        random.normal(d_key, (n_out, nd)),
        random.normal(w_key, (n_out, nw)),
    )


def init_network_params(
    n_outs: List[int], key: Array, nx: int, nd: int, nw: int
) -> List[Array]:
    keys = random.split(key, len(n_outs))
    return [
        init_layer_params(nx, nd, nw, n_out, key) for n_out, key in zip(n_outs, keys)
    ]


@jit
def update(
    theta: ArrayLike,
    f: Callable[[ArrayLike], Array],
    df: Callable[[ArrayLike], Array],
    d: ArrayLike,
    e: ArrayLike,
    s=0.01,
) -> List[Array]:
    return [
        (x_param - s * d_x_param, d_param - s * d_d_parm, w_param - s * d_w_param)
        for (x_param, d_param, w_param), (d_x_param, d_d_parm, d_w_param) in zip(
            theta, df(theta, d, e)
        )
    ]
