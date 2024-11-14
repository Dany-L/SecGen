from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Type, Union

import numpy as np
import torch
from pydantic import BaseModel

from .configuration.base import InputOutput
from .models.base import DynamicIdentificationModel
from .models.recurrent import BasicLstm, BasicRnn
from .tracker import events as ev
from .tracker.base import AggregatedTracker


class AdditionalTestConfig(BaseModel):
    epochs: int
    horizon: int
    sampling_type: Literal["optimize", "random"]
    lr: float = 0.01


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
        model: DynamicIdentificationModel,
        tracker: AggregatedTracker = AggregatedTracker(),
    ):
        self.epochs = config.epochs
        self.horizon = config.horizon
        self.lr = config.lr
        self.model = model
        self.nx, self.ne, self.nd, self.B = model.nx, model.ne, model.nd, 1
        self.sampling_type = config.sampling_type
        self.tracker = tracker

    @abstractmethod
    def test(self, model: DynamicIdentificationModel) -> AdditionalTestResult:
        pass


class StabilityOfInitialState(AdditionalTest):

    def test(self, model: DynamicIdentificationModel) -> AdditionalTestResult:
        d = torch.zeros(1, self.horizon, self.nd)
        xk_norm_max, e_hat_max, x0_max = (
            torch.inf,
            np.zeros((self.horizon, self.ne)),
            np.zeros((self.nx, 1)),
        )

        self.tracker.track(
            ev.Log(
                "",
                f"Testing stability of initial state, sampling type: {self.sampling_type}",
            )
        )
        x0 = self.get_x0()
        if self.sampling_type == "optimize":
            for x0_i in x0:
                x0_i.requires_grad = True
            opt = torch.optim.Adam(x0, lr=self.lr, maximize=True)

        for epoch in range(self.epochs):
            e_hat, x = model.forward(d, x0)
            xk = self.get_xk(x)
            xk_norm = torch.linalg.norm(xk, ord=2)
            if self.sampling_type == "optimize":
                xk_norm.backward(retain_graph=True)
                opt.step()
                opt.zero_grad()
            elif self.sampling_type == "random":
                x0 = self.get_x0()
            self.tracker.track(
                ev.Log(
                    "",
                    f"{epoch}/{self.epochs}: xk norm: {xk_norm.cpu().detach().numpy()}",
                )
            )

            if xk_norm < xk_norm_max:
                xk_norm_max, x0_max, e_hat_max = (
                    xk_norm,
                    x0,
                    e_hat[0, :, :].cpu().detach().numpy(),
                )

        return AdditionalTestResult(
            float(xk_norm_max.cpu().detach().numpy()),
            [InputOutput(d=d, e_hat=e_hat_max)],
            {
                "x0": np.hstack(
                    [x0_i.cpu().detach().numpy() for x0_i in x0_max]
                ).tolist()
            },
        )

    def get_x0(self) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(self.model, BasicLstm):
            x0 = (
                torch.rand(self.model.num_layers, self.B, self.nx),
                torch.rand(self.model.num_layers, self.B, self.nx),
            )
        elif isinstance(self.model, BasicRnn):
            x0 = (torch.rand(self.model.num_layers, self.B, self.nx),)
        else:
            x0 = (torch.rand(self.B, self.nx),)
        return x0

    def get_xk(
        self, x: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(self.model, BasicLstm):
            xk = torch.hstack([h[-1, 0, :] for h in x]).reshape(2 * self.nx, 1)
        elif isinstance(self.model, BasicRnn):
            xk = x[0][-1, 0, :].reshape(self.nx, 1)
        else:
            xk = x[0][0, :].reshape(self.nx, 1)
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
