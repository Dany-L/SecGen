from typing import Callable, List, Literal, Optional, Tuple, Union

import torch

from .base import DynamicIdentificationConfig, DynamicIdentificationModel


class BasicRnnConfig(DynamicIdentificationConfig):
    num_layers: int = 5
    nonlinearity: Literal["tanh", "relu"]


class BasicRnn(DynamicIdentificationModel):
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

    def add_semidefinite_constraints(self, constraints=List[Callable]) -> None:
        pass

    def add_pointwise_constraints(self, constraints=List[Callable]) -> None:
        pass

    def initialize_parameters(self) -> None:
        pass

    def project_parameters(self) -> None:
        pass


class BasicLstmConfig(DynamicIdentificationConfig):
    dropout: float = 0.25
    num_layers: int = 5


class BasicLstm(DynamicIdentificationModel):
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
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
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
