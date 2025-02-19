from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pydantic import BaseModel

from .configuration.base import InputOutput
from .models.torch import base as base_torch
from .models.torch import recurrent as recurrent_torch
from .tracker import events as ev
from .tracker.base import AggregatedTracker
from .utils.base import get_sequence_norm


class AdditionalTestConfig(BaseModel):
    epochs: int
    horizon: int
    sampling_type: Literal["optimize", "random"] = "optimize"
    lr: float = 0.01
    scale: float = 1.0


@dataclass
class AdditionalTestResult:
    value: float
    input_output: List[InputOutput]
    additional: Dict[str, Union[List[float], float]]


class AdditionalTest:
    CONFIG: Type[AdditionalTestConfig] = AdditionalTestConfig

    def __init__(
        self,
        config: AdditionalTestConfig,
        model: base_torch.ConstrainedModule,
        tracker: AggregatedTracker = AggregatedTracker(),
    ):
        self.epochs = config.epochs
        self.horizon = config.horizon
        self.lr = config.lr
        self.model = model
        self.nx, self.ne, self.nd, self.B = model.nx, model.ne, model.nd, 1
        self.sampling_type = config.sampling_type
        self.tracker = tracker
        self.scale = config.scale

    @abstractmethod
    def test(self) -> AdditionalTestResult:
        pass


class StabilityOfInitialState(AdditionalTest):

    def test(self) -> AdditionalTestResult:
        d = torch.zeros(1, self.horizon, self.nd)
        xk_norm_max, e_hat_max, x0_max = (
            torch.tensor(0.0),
            torch.zeros((1, self.horizon, self.ne)),
            (torch.zeros((self.nx, 1))),
        )

        x0 = get_x0(self.model, self.B)
        if self.sampling_type == "optimize":
            for x0_i in x0:
                x0_i.requires_grad = True
            opt = torch.optim.Adam(x0, lr=self.lr, maximize=True)
        for epoch in range(self.epochs):
            e_hat, x = self.model.forward(d, x0)
            xk = get_xk(self.model, x)
            xk_norm = torch.linalg.norm(xk, ord=2)
            if self.sampling_type == "optimize":
                xk_norm.backward(retain_graph=True)
                opt.step()
                opt.zero_grad()
            elif self.sampling_type == "random":
                x0 = get_x0(self.model, self.B)

            if xk_norm > xk_norm_max:
                xk_norm_max, x0_max, e_hat_max = (
                    xk_norm.clone(),
                    [x0_i.clone() for x0_i in x0],
                    e_hat.clone(),
                )
                self.tracker.track(
                    ev.Log(
                        "",
                        f"{epoch}/{self.epochs}: xk norm: {xk_norm.cpu().detach().numpy():.2f}, e_hat norm {torch.linalg.norm(e_hat).cpu().detach().numpy():.2f}",
                    )
                )

        return AdditionalTestResult(
            float(xk_norm_max.cpu().detach().numpy()),
            [
                InputOutput(
                    d=d.cpu().detach().numpy(),
                    e_hat=e_hat_max.cpu().detach().numpy(),
                    x0=np.array([x0_i.cpu().detach().numpy() for x0_i in x0_max]),
                )
            ],
            {},
        )


class InputOutputStabilityL2(AdditionalTest):
    def test(
        self,
        x0: Optional[
            Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> AdditionalTestResult:
        d = self.scale * torch.randn(self.B, self.horizon, self.nd)
        d.requires_grad = True
        ga_2_max, e_hat_max, d_max = (
            torch.tensor(0.0),
            torch.zeros((self.horizon, self.ne)),
            torch.zeros((self.horizon, self.nd)),
        )
        opt = torch.optim.Adam([d], lr=self.lr, maximize=True)
        for epoch in range(self.epochs):
            opt.zero_grad()
            e_hat, _ = self.model.forward(d, x0)
            ga_2 = get_sequence_norm(e_hat) / get_sequence_norm(d)
            ga_2.backward(retain_graph=True)
            opt.step()

            if ga_2 > ga_2_max:
                ga_2_max, e_hat_max, d_max = (
                    ga_2.clone().detach(),
                    e_hat.clone().detach(),
                    d.clone().detach(),
                )
            if epoch % 100 == 0:
                self.tracker.track(
                    ev.Log(
                        "",
                        f"{epoch}/{self.epochs}: ga: {np.sqrt(ga_2.cpu().detach().numpy()):.2f}, norm e_hat: {torch.linalg.norm(e_hat).cpu().detach().numpy():.2f}, norm d: {torch.linalg.norm(d).cpu().detach().numpy():.2f}",
                    )
                )

        return AdditionalTestResult(
            np.sqrt(ga_2_max.cpu().detach().numpy()),
            [
                InputOutput(
                    d=d_max.cpu().detach().numpy(),
                    e_hat=e_hat_max.cpu().detach().numpy(),
                )
            ],
            {},
        )


def get_x0(
    model: base_torch.ConstrainedModule, B: int
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(model, recurrent_torch.BasicLstm):
        x0 = (
            torch.rand(model.num_layers, B, model.nx),
            torch.rand(model.num_layers, B, model.nx),
        )
    elif isinstance(model, recurrent_torch.BasicRnn):
        x0 = (torch.rand(model.num_layers, B, model.nx),)
    else:
        x0 = (torch.rand(B, model.nx),)
    return x0


def get_xk(
    model: base_torch.ConstrainedModule,
    x: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
) -> torch.Tensor:
    if isinstance(model, recurrent_torch.BasicLstm):
        xk = torch.hstack([h[-1, 0, :] for h in x]).reshape(2 * model.nx, 1)
    elif isinstance(model, recurrent_torch.BasicRnn):
        xk = x[0][-1, 0, :].reshape(model.nx, 1)
    else:
        xk = x[0][0, :].reshape(model.nx, 1)
    return xk


def retrieve_additional_test_class(metric_class_string: str) -> Type[AdditionalTest]:
    # https://stackoverflow.com/a/452981
    parts = metric_class_string.split(".")
    module_string = ".".join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, AdditionalTest):
        raise ValueError(f"{cls} is not a subclass of Metrics.")
    return cls  # type: ignore
