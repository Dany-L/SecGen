import dataclasses
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .configuration import (FIG_FOLDER_NAME, DynamicIdentificationConfig,
                            ExperimentBaseConfig, NormalizationParameters)
from .data_io import (save_model, save_model_parameter, save_sequences_to_mat,
                      write_config)
from .models.base import ConstrainedModule
from .utils import base as utils
from .utils import plot


@dataclasses.dataclass
class Event:
    msg: str


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
    e_hats: List[NDArray[np.float64]]
    es: List[NDArray[np.float64]]
    file_name: str


@dataclasses.dataclass
class SaveModelParameter(ModelEvent):
    pass


@dataclasses.dataclass
class SaveConfig(Event):
    name: str
    config: Union[ExperimentBaseConfig, DynamicIdentificationConfig]


@dataclasses.dataclass
class SaveNormalization(Event):
    input: NormalizationParameters
    output: NormalizationParameters


@dataclasses.dataclass
class SaveEvaluation(Event):
    results: Dict[str, float]
    mode: Literal["test", "validation"]


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
        elif isinstance(event, SaveNormalization):
            np.savez(
                os.path.join(self.directory, "normalization.npz"),
                input_mean=event.input.mean,
                input_std=event.input.std,
                output_mean=event.output.mean,
                output_std=event.output.std,
            )
            self.write_to_logfile(
                f"Save normalization mean and std to {self.directory}"
            )
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
            save_model(
                event.model,
                self.directory,
                utils.get_model_file_name(event.name, self.model_name),
            )
            self.write_to_logfile(
                f"Save model to {utils.get_model_file_name(event.name, self.model_name)} in {self.directory}"
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
                os.path.join(
                    self.directory,
                    utils.get_config_file_name(event.name, self.model_name),
                ),
            )
            self.write_to_logfile(f"Save model config json file to {self.directory}")
        elif isinstance(event, SaveEvaluation):
            with open(
                os.path.join(
                    self.directory, f"evaluate-{self.model_name}-{event.mode}.json"
                ),
                "w",
            ) as f:
                json.dump(event.results, f)
            self.write_to_logfile(f"Save evaluation results to {self.directory}")
        else:
            raise ValueError(f"Event is not defined {event}")

    def write_to_logfile(self, msg: str) -> None:
        with open(self.log_file_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {msg}\n")
