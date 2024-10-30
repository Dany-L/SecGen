from typing import Optional

from .. import tracker as base_tracker
from ..configuration import BaseConfig
from .base import ConstrainedModule
from .sector_bounded import GeneralSectorBoundedLtiRnn, SectorBoundedLtiRnn


def get_model_from_config(
    model_name: str,
    base_config: BaseConfig,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> ConstrainedModule:
    model: Optional[ConstrainedModule] = None
    if model_name == "tanh":
        model = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="tanh",
            tracker=tracker,
        )
    elif model_name == "dzn":
        model = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="deadzone",
            tracker=tracker,
        )
    elif model_name == "dznGen":
        model = GeneralSectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="deadzone",
            tracker=tracker,
        )
    else:
        ValueError(f"Model name {model_name} is not implemented.")

    return model
