from typing import Iterable, List, Tuple

import torch
import torch.optim as opt

from .configuration.experiment import BaseExperimentConfig
from .models import base
from .models import base_jax
from .models import base_torch


def get_optimizer(
    config: BaseExperimentConfig, models: Tuple[base.DynamicIdentificationModel]
) -> List[opt.Optimizer]:
    if isinstance(models[0], base_torch.DynamicIdentificationModel):
        params = (m.parameters() for m in models)
        opt_config = config.optimizer
        initial_hidden_state = config.initial_hidden_state
        if initial_hidden_state == "joint":
            params_list = []
            for param in params:
                params_list += list(param)
            if opt_config.name == "adam":
                optimizers = [opt.Adam(params_list, lr=opt_config.learning_rate)]
            elif opt_config.name == "sgd":
                optimizers = [opt.SGD(params_list, lr=opt_config.learning_rate)]
            else:
                raise ValueError(f"Unsupported optimizer name: {opt_config.name}")
        else:  # same behavior for zero and separate initialization
            if opt_config.name == "adam":
                optimizers = [
                    opt.Adam(param, lr=opt_config.learning_rate) for param in params
                ]
            elif opt_config.name == "sgd":
                optimizers = [
                    opt.SGD(param, lr=opt_config.learning_rate) for param in params
                ]
            else:
                raise ValueError(f"Unsupported optimizer name: {opt_config.name}")
        return optimizers
    else:
        if config.initial_hidden_state == "joint":
            return [None]
        else:
            return [None, None]
