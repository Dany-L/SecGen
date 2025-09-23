from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from ..utils import transformation as trans
from . import base

class BasicS4Config(base.DynamicIdentificationConfig):
    num_layers: int = 1

class BasicS4(base.DynamicIdentificationModel):
    CONFIG = BasicS4Config

    def __init__(
        self,
        config: BasicS4Config,
    ) -> None:
        # based on https://doi.org/10.1016/j.ifacol.2024.08.536
        super().__init__(config)
        from ssm.StackedSSM import StackedSSMModel
        from ssm.layer import LRU, S4, S5

        s4_kwargs = {"dt_range":(1.0, 0.1), "order":2}

        # if we have skip_connection true then there is an implicit constraint that hidden_units == in_features, otherwise there is a mismatch in dimensions. However, it works for inputs with size one because then torch justs adds the dimension to each dimension of the hidden unit. Probably this is not intended behaviour.
        self.s4 = StackedSSMModel(
            in_features=config.nd,
            hidden_units=config.nd,
            out_features=config.ne,
            state_sizes=[config.nz]*config.num_layers,
            base_model_kwargs=s4_kwargs,
            base_model=S4,
            activation_fnc=nn.ELU(),
            skip_connection=True
        )

    def forward(
        self,
        d: torch.Tensor,
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
        e_hat, x = self.s4(d)
        # e_hat, h = self.s4(d, x0)
        return (e_hat, (x,))