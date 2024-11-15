import dataclasses
from typing import Any, Dict, List, Literal, Optional, Union

from matplotlib.figure import Figure

from ..configuration.base import InputOutput, NormalizationParameters
from ..models.base import ConstrainedModule
from .base import BaseTrackerConfig, Event


@dataclasses.dataclass
class Log(Event):
    log_msg: str


@dataclasses.dataclass
class SaveFig(Event):
    fig: Figure
    name: str


@dataclasses.dataclass
class Start(Event):
    dataset_name: str


@dataclasses.dataclass
class SaveTrackingConfiguration(Event):
    config: Dict[str, BaseTrackerConfig]
    model_name: str
    model_directory: str


@dataclasses.dataclass
class LoadTrackingConfiguration(Event):
    model_directory: str
    model_name: str


@dataclasses.dataclass
class Stop(Event):
    pass


@dataclasses.dataclass
class SetTags(Event):
    tags: Dict[str, Union[str, bool]]


@dataclasses.dataclass
class TrackMetrics(Event):
    metrics: Dict[str, float]
    step: Optional[int] = None


@dataclasses.dataclass
class ModelEvent(Event):
    model: ConstrainedModule


@dataclasses.dataclass
class SaveModel(ModelEvent):
    name: str


@dataclasses.dataclass
class SaveSequences(Event):
    sequences: List[InputOutput]
    file_name: str


@dataclasses.dataclass
class SaveModelParameter(ModelEvent):
    pass


@dataclasses.dataclass
class TrackParameters(Event):
    name: str
    parameters: Dict[str, Any]


@dataclasses.dataclass
class SaveNormalization(Event):
    input: NormalizationParameters
    output: NormalizationParameters


@dataclasses.dataclass
class SaveEvaluation(Event):
    results: Dict[str, Any]
    mode: Literal["test", "validation"]
