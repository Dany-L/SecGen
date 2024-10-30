from .sector_bounded import SectorBoundedLtiRnn, GeneralSectorBoundedLtiRnn
from .base import ConstrainedModule
from ..configuration import BaseConfig
from .. import tracker as base_tracker
from typing import Optional

def get_model_from_config(model_name:str, base_config:BaseConfig,tracker: Optional[base_tracker.BaseTracker] = base_tracker.BaseTracker()) -> ConstrainedModule:
    if model_name =='tanh':
        model = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity='tanh',
            tracker=tracker
        )
    elif model_name == 'dzn':
        model = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity='deadzone',
            tracker=tracker
        )
    elif model_name == 'dznGen':
        model = GeneralSectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity='deadzone',
            tracker=tracker
        )
    else:
        ValueError(f"Model name {model_name} is not implemented.")

    return model