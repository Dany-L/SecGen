import json
import os
import time
from datetime import datetime
from typing import Literal

from ..configuration.base import FIG_FOLDER_NAME
from ..data_io import (
    save_model,
    save_model_parameter,
    save_sequences_to_mat,
    write_config,
)
from ..utils import base as utils
from ..utils import plot
from . import events as ev
from .base import Event


class IoTracker:
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
        if isinstance(event, ev.Log):
            print(event.log_msg)
            self.write_to_logfile(event.log_msg)
        elif isinstance(event, ev.SaveNormalization):
            with open(
                os.path.join(self.directory, "normalization.json"),
                "w",
            ) as f:
                json.dump(
                    dict(
                        input_mean=float(event.input.mean),
                        input_std=float(event.input.std),
                        output_mean=float(event.output.mean),
                        output_std=float(event.output.std),
                    ),
                    f,
                )
            self.write_to_logfile(
                f"Save normalization mean and std to {self.directory}"
            )
        elif isinstance(event, ev.SaveFig):
            fig_subdirectory = os.path.join(self.directory, FIG_FOLDER_NAME)
            os.makedirs(fig_subdirectory, exist_ok=True)
            plot.save_fig(event.fig, event.name, fig_subdirectory)
            self.write_to_logfile(f"save fig {event.name} in {self.directory}")
        elif isinstance(event, ev.Start):
            self.start_time = time.time()
            self.write_to_logfile(f"--- Start model {self.model_name} ---")
        elif isinstance(event, ev.Stop):
            self.write_to_logfile(
                f"--- Stop duration: {utils.get_duration_str(self.start_time, time.time())} ---"
            )
        elif isinstance(event, ev.SaveModel):
            save_model(
                event.model,
                self.directory,
                utils.get_model_file_name(event.name, self.model_name),
            )
            self.write_to_logfile(
                f"Save model to {utils.get_model_file_name(event.name, self.model_name)} in {self.directory}"
            )
        elif isinstance(event, ev.SaveModelParameter):
            save_model_parameter(
                event.model,
                os.path.join(self.directory, f"model_params-{self.model_name}.mat"),
            )
            self.write_to_logfile(
                f"Save model parameters as mat file to {self.directory}"
            )
        elif isinstance(event, ev.SaveSequences):
            save_sequences_to_mat(
                event.sequences, os.path.join(self.directory, event.file_name)
            )
            self.write_to_logfile(
                f"Save sequences to {event.file_name} in {self.directory}"
            )
        elif isinstance(event, ev.SaveConfig):
            write_config(
                event.config,
                os.path.join(
                    self.directory,
                    utils.get_config_file_name(event.name, self.model_name),
                ),
            )
            self.write_to_logfile(f"Save model config json file to {self.directory}")
        elif isinstance(event, ev.SaveEvaluation):
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
