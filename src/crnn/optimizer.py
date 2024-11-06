from typing import Iterable, Optional, Tuple

import torch
import torch.optim as opt

from .configuration import OptimizerConfig


def get_optimizer(
    opt_config: OptimizerConfig, params: Tuple[Iterable[torch.Tensor]]
) -> Tuple[Optional[opt.Optimizer]]:
    optimizers: Optional[Tuple[opt.Optimizer]] = None
    if opt_config.initial_hidden_state == "joint":
        params_list = []
        for param in params:
            params_list += list(param)
        if opt_config.name == "adam":
            optimizers = (None, opt.Adam(params_list, lr=opt_config.learning_rate))
        elif opt_config.name == "sgd":
            optimizers = (None, opt.SGD(params_list, lr=opt_config.learning_rate))
        else:
            raise ValueError(f"Unsupported optimizer name: {opt_config.name}")
    else:  # same behavior for zero and separate initialization
        if opt_config.name == "adam":
            optimizers = (
                opt.Adam(param, lr=opt_config.learning_rate) for param in params
            )
        elif opt_config.name == "sgd":
            optimizers = (
                opt.SGD(param, lr=opt_config.learning_rate) for param in params
            )
        else:
            raise ValueError(f"Unsupported optimizer name: {opt_config.name}")
    return optimizers
