from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jax.typing import ArrayLike
from numpy.typing import NDArray

from ...utils import transformation as trans
from .. import base
from . import base as base_torch


class BasicRnnConfig(base_torch.DynamicIdentificationConfig):
    num_layers: int = 5
    nonlinearity: Literal["tanh", "relu"]


class BasicRnn(base_torch.DynamicIdentificationModel):
    CONFIG = BasicRnnConfig

    def __init__(
        self,
        config: BasicRnnConfig,
        # tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
    ) -> None:
        super().__init__(config)
        self.num_layers = config.num_layers
        self.rnn_layer = torch.nn.RNN(
            input_size=config.nd,
            hidden_size=config.nz,
            nonlinearity=config.nonlinearity,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.output_layer = torch.nn.Linear(
            in_features=config.nz, out_features=config.ne
        )
        # self.tracker = tracker

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
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        if x0 is None:
            x0 = torch.zeros(size=(self.num_layers, B, self.nx))
        else:
            x0 = x0[0]
        x, h = self.rnn_layer.forward(d, x0)
        e_hat = self.output_layer.forward(x)
        return (e_hat, (h,))

    def project_parameters(self) -> None:
        pass


class BasicLstmConfig(base_torch.DynamicIdentificationConfig):
    dropout: float = 0.25
    num_layers: int = 5


class BasicLstm(base_torch.DynamicIdentificationModel):
    CONFIG = BasicLstmConfig

    def __init__(self, config: BasicLstmConfig) -> None:
        super().__init__(config)
        self.num_layers = config.num_layers
        self.lstm_layer = torch.nn.LSTM(
            input_size=config.nd,
            hidden_size=config.nz,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.output_layer = torch.nn.Linear(in_features=self.nz, out_features=self.ne)
        # self.tracker = tracker
        for name, param in self.lstm_layer.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_normal_(param)

        torch.nn.init.xavier_normal_(self.output_layer.weight)

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
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        x, (h, c) = self.lstm_layer.forward(d, x0)
        e_hat = self.output_layer.forward(x)
        return (e_hat, (h, c))

    def project_parameters(self) -> None:
        pass


class BasicLtiRnn(base_torch.ConstrainedModule):
    CONFIG = base_torch.ConstrainedModuleConfig

    def __init__(self, config: base_torch.ConstrainedModuleConfig) -> None:
        super().__init__(config)
        self.nonlinearity = config.nonlinearity

        mean = 0.0
        self.A = torch.zeros((self.nx, self.nx))
        self.B = torch.zeros((self.nx, self.nd))
        self.B2 = torch.eye(self.nw)

        self.C = torch.zeros((self.ne, self.nx))
        self.D = torch.zeros((self.ne, self.nd))

        self.D12 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.ne, self.nw)),
                1 / self.ne * torch.ones((self.ne, self.nw)),
            )
        )
        self.C2 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nx)),
                1 / self.nz * torch.ones((self.nz, self.nx)),
            )
        )
        self.D21 = torch.nn.Parameter(
            torch.normal(
                mean * torch.ones((self.nz, self.nd)),
                1 / self.nz * torch.ones((self.nz, self.nd)),
            )
        )
        self.D22 = torch.zeros((self.nz, self.nw))

    def set_lure_system(self) -> base.LureSystemClass:
        theta = trans.torch_bmat(
            [
                [self.A, self.B, self.B2],
                [self.C, self.D, self.D12],
                [self.C2, self.D21, self.D22],
            ]
        )
        sys = base.get_lure_matrices(theta, self.nx, self.nd, self.ne, self.nl)
        self.lure = base.LureSystem(sys)

        return sys

    def check_constraints(self) -> bool:
        return True

    def project_parameters(self) -> None:
        pass
