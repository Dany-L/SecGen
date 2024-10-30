import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .configuration import FIG_FOLDER_NAME, BaseConfig
from .io import (save_model, save_model_parameter, save_sequences_to_mat,
                 write_config)
from .models.base import ConstrainedModule
from .utils import base as utils
from .utils import plot


@dataclass
class Event:
    msg: str


@dataclass
class Log(Event):
    log_msg: str


@dataclass
class SaveFig(Event):
    fig: Figure
    name: str


@dataclass
class Start(Event):
    pass


@dataclass
class Stop(Event):
    pass


@dataclass
class ModelEvent(Event):
    model: ConstrainedModule


@dataclass
class SaveModel(ModelEvent):
    pass


@dataclass
class SaveSequences(Event):
    e_hats: List[NDArray[np.float64]]
    es: List[NDArray[np.float64]]
    file_name: str


@dataclass
class SaveModelParameter(ModelEvent):
    pass


@dataclass
class SaveConfig(Event):
    config: BaseConfig


class BaseTracker:
    def __init__(
        self,
        directory: str = os.environ["HOME"],
        model_name: str = "",
        type: Literal["training", "validation"] = "training",
    ) -> None:
        self.directory = directory
        self.model_name = model_name
        self.log_file_path = str(os.path.join(self.directory, f"{type}.log"))

    def track(self, event: Event) -> None:
        if isinstance(event, Log):
            print(event.log_msg)
            self.write_to_logfile(event.log_msg)
        elif isinstance(event, SaveFig):
            fig_subdirectory = os.path.join(self.directory, FIG_FOLDER_NAME)
            os.makedirs(fig_subdirectory, exist_ok=True)
            plot.save_fig(event.fig, event.name, fig_subdirectory)
            self.write_to_logfile(f"save fig {event.name} in {self.directory}")
        elif isinstance(event, Start):
            self.start_time = time.time()
            self.write_to_logfile(f"--- Start model {self.model_name} ---")
        elif isinstance(event, Stop):
            self.write_to_logfile(
                f"--- Stop duration: {utils.get_duration_str(self.start_time, time.time())} ---"
            )
        elif isinstance(event, SaveModel):
            save_model(event.model, self.directory, self.get_model_file_name())
            self.write_to_logfile(
                f"Save model to {self.get_model_file_name()} in {self.directory}"
            )
        elif isinstance(event, SaveModelParameter):
            save_model_parameter(
                event.model,
                os.path.join(self.directory, f"model_params-{self.model_name}.mat"),
            )
            self.write_to_logfile(
                f"Save model parameters as mat file to {self.directory}"
            )
        elif isinstance(event, SaveSequences):
            save_sequences_to_mat(
                event.e_hats, event.es, os.path.join(self.directory, event.file_name)
            )
            self.write_to_logfile(
                f"Save sequences to {event.file_name} in {self.directory}"
            )
        elif isinstance(event, SaveConfig):
            write_config(
                event.config,
                os.path.join(self.directory, f"config-{self.model_name}.json"),
            )
            self.write_to_logfile(f"Save model config json file to {self.directory}")
        else:
            raise ValueError(f"Event is not defined {event}")

    def write_to_logfile(self, msg: str) -> None:
        with open(self.log_file_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {msg}\n")

    def get_model_file_name(self) -> str:
        return f"parameters-{self.model_name}.pth"
