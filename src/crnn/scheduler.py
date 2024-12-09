from typing import Iterable, List, Tuple

import torch
import torch.optim as opt

from .configuration.experiment import BaseExperimentConfig
from .models import base
from .models import base_jax
from torch.optim import lr_scheduler


def get_scheduler(
    optimizers: Tuple[base.DynamicIdentificationModel],
) -> List[lr_scheduler.ReduceLROnPlateau]:
    if optimizers[0] is None:
        return [None for _ in optimizers]
    else:
        return [
            lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)
            for optimizer in optimizers
        ]
