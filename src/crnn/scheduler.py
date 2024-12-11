from typing import List, Tuple

from torch.optim import lr_scheduler

from .models import base


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
