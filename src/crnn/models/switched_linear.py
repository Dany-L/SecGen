from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from deepsysid.networks.switching import UnconstrainedSwitchingLSTM

from ..utils import transformation as trans
from . import base

class ReliNetConf(base.DynamicIdentificationConfig):
    dropout: float = 0.25
    num_layers: int = 1
    

class ReliNet(base.DynamicIdentificationModel):
    CONFIG = ReliNetConf

    def __init__(
        self,
        config: ReliNetConf,
    ) -> None:
        # based on add reference to ReLiNet
        super().__init__(config)

        self.relinet = UnconstrainedSwitchingLSTM(
            control_dim=config.nd,
            state_dim=config.nz,
            output_dim=config.ne,
            recurrent_dim=config.nz,
            num_recurrent_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        d: torch.Tensor,
        previous_e: Optional[torch.Tensor] = None,
        x0: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
        ] = None,
        theta: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        B, N, nu = d.shape  # number of batches, length of sequence, input size
        if x0 is None:
            x0 = torch.zeros(size=(B, self.nz))
        else:
            x0 = x0[0].squeeze(0)
        outputs= self.relinet(d,previous_e, x0[-1])
        # e_hat, h = self.s4(d, x0)
        return (outputs.outputs, (outputs.states,))

