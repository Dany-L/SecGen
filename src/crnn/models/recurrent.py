from typing import Callable, List, Literal, Optional, Tuple, Union

import torch

from .. import tracker as base_tracker
from .base import DynamicIdentificationModel


class BasicRnn(DynamicIdentificationModel):
    def __init__(
        self,
        nd: int,
        ne: int,
        nz: int,
        num_layers: int = 5,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
    ) -> None:
        super().__init__(nd, ne, nz, nonlinearity)
        self.num_layers = num_layers
        self.rnn_layer = torch.nn.RNN(
            input_size=nd,
            hidden_size=nz,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.output_layer = torch.nn.Linear(in_features=nz, out_features=ne)
        self.tracker = tracker

    def forward(
        self, d: torch.Tensor, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        if x0 is None:
            x0 = torch.zeros(size=(self.num_layers, B, self.nx))
        x, h = self.rnn_layer.forward(d, x0)
        e_hat = self.output_layer.forward(x)
        return (e_hat, h)

    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    def initialize_parameters(self) -> None:
        pass

    def project_parameters(self) -> None:
        pass


class BasicLstm(DynamicIdentificationModel):
    def __init__(
        self,
        nd: int,
        ne: int,
        nz: int,
        num_layers: int = 5,
        dropout: float = 0.25,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
    ) -> None:
        super().__init__(nd, ne, nz, nonlinearity)
        self.num_layers = num_layers
        self.lstm_layer = torch.nn.LSTM(
            input_size=nd,
            hidden_size=nz,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = torch.nn.Linear(in_features=nz, out_features=ne)
        self.tracker = tracker
        for name, param in self.lstm_layer.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_normal_(param)

        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(
        self, d: torch.Tensor, x0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        x, (h, c) = self.lstm_layer.forward(d, x0)
        e_hat = self.output_layer.forward(x)
        return (e_hat, (h, c))

    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    def initialize_parameters(self) -> None:
        pass

    def project_parameters(self) -> None:
        pass
