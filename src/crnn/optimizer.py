from typing import Iterable, Optional

import torch
import torch.optim as opt

from .configuration import OptimizerConfig


def get_optimizer(
    opt_config: OptimizerConfig, params: Iterable[torch.Tensor]
) -> opt.Optimizer:
    optimizer: Optional[opt.Optimizer] = None
    if opt_config.name == "adam":
        optimizer = opt.Adam(params, lr=opt_config.learning_rate)
    elif opt_config.name == "sgd":
        optimizer = opt.SGD(params, lr=opt_config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer name: {opt_config.name}")
    return optimizer
