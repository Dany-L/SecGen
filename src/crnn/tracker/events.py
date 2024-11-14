import dataclasses
from typing import Any, Dict, List, Literal

from matplotlib.figure import Figure
from pydantic import BaseModel

from ..configuration.base import InputOutput, NormalizationParameters
from ..models.base import ConstrainedModule
from .base import Event


@dataclasses.dataclass
class Log(Event):
    log_msg: str


@dataclasses.dataclass
class SaveFig(Event):
    fig: Figure
    name: str


@dataclasses.dataclass
class Start(Event):
    pass


@dataclasses.dataclass
class Stop(Event):
    pass


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
class SaveConfig(Event):
    name: str
    config: BaseModel


@dataclasses.dataclass
class SaveNormalization(Event):
    input: NormalizationParameters
    output: NormalizationParameters


@dataclasses.dataclass
class SaveEvaluation(Event):
    results: Dict[str, Any]
    mode: Literal["test", "validation"]
