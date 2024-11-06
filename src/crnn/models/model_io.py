from typing import Optional, Tuple

from .. import tracker as base_tracker
from ..configuration import BaseConfig
from .base import ConstrainedModule
from .recurrent import BasicLstm, BasicRnn
from .sector_bounded import GeneralSectorBoundedLtiRnn, SectorBoundedLtiRnn


def get_model_from_config(
    model_name: str,
    base_config: BaseConfig,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[ConstrainedModule]:
    predictor: Optional[ConstrainedModule] = None
    initializer: Optional[ConstrainedModule] = None
    if model_name == "tanh":
        initializer = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="tanh",
            tracker=tracker,
        )
        predictor = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="tanh",
            tracker=tracker,
        )
    elif model_name == "dzn":
        initializer = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="deadzone",
            tracker=tracker,
        )
        predictor = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="deadzone",
            tracker=tracker,
        )
    elif model_name == "sat":
        initializer = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="sat",
            tracker=tracker,
        )
        predictor = SectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="sat",
            tracker=tracker,
        )
    elif model_name == "dznGen":
        initializer = GeneralSectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="deadzone",
            tracker=tracker,
        )
        predictor = GeneralSectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="deadzone",
            tracker=tracker,
        )
    elif model_name == "satGen":
        initializer = GeneralSectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="sat",
            tracker=tracker,
        )
        predictor = GeneralSectorBoundedLtiRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            nonlinearity="sat",
            tracker=tracker,
        )
    elif model_name == "rnn":
        initializer = BasicRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            num_layers=base_config.num_layer,
            nonlinearity="tanh",
            tracker=tracker,
        )
        predictor = BasicRnn(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            num_layers=base_config.num_layer,
            nonlinearity="tanh",
            tracker=tracker,
        )
    elif model_name == "lstm":
        initializer = BasicLstm(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            num_layers=base_config.num_layer,
            nonlinearity="tanh",
            dropout=0.25,
            tracker=tracker,
        )
        predictor = BasicLstm(
            nz=base_config.nz,
            nd=len(base_config.input_names),
            ne=len(base_config.output_names),
            num_layers=base_config.num_layer,
            nonlinearity="tanh",
            dropout=0.25,
            tracker=tracker,
        )
    else:
        ValueError(f"Model name {model_name} is not implemented.")

    return (initializer, predictor)
