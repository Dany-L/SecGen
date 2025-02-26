import dataclasses
from typing import Any, Dict, List, Literal, Optional, Union

from matplotlib.figure import Figure

from ..configuration.base import InputOutput, NormalizationParameters
from ..models.base import DynamicIdentificationModel, Linear
from .base import Event, TrackerConfig


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
    config: Dict[str, TrackerConfig]
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
    model: DynamicIdentificationModel


@dataclasses.dataclass
class SaveModel(ModelEvent):
    name: str


@dataclasses.dataclass
class SaveSequences(Event):
    sequences: List[InputOutput]
    file_name: str


@dataclasses.dataclass
class SaveModelParameter(ModelEvent):
    name_suffix: str = ''


@dataclasses.dataclass
class TrackParameters(Event):
    name: str
    parameters: Dict[str, Any]


@dataclasses.dataclass
class TrackResults(Event):
    name: str
    parameters: Dict[str, Any]


@dataclasses.dataclass
class TrackParameter(Event):
    name: str
    parameter: str


@dataclasses.dataclass
class SaveNormalization(Event):
    input: NormalizationParameters
    output: NormalizationParameters


@dataclasses.dataclass
class SaveInitialization(Event):
    ss: Linear
    data: Dict[str, Any]


@dataclasses.dataclass
class SaveEvaluation(Event):
    results: Dict[str, Any]
    mode: Literal["test", "validation"]
