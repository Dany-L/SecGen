import json
import os
import time
from datetime import datetime

from ..configuration.base import FIG_FOLDER_NAME, SEQ_FOLDER_NAME
from ..data_io import (save_input_output_to_mat, save_model,
                       save_model_parameter, write_dict_to_json)
from ..utils import base as utils
from ..utils import plot
from . import events as ev
from .base import BaseTracker, BaseTrackerConfig, Event


class IoTrackerConfig(BaseTrackerConfig):
    pass


class IoTracker(BaseTracker):
    CONFIG = IoTrackerConfig

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
            self.write_to_logfile(f"--- Start model {self.model_name} ---")
        elif isinstance(event, ev.Stop):
            self.write_to_logfile(f"--- Stop ---")
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
            seq_subdirectory = os.path.join(self.directory, SEQ_FOLDER_NAME)
            os.makedirs(seq_subdirectory, exist_ok=True)
            save_input_output_to_mat(
                event.sequences, os.path.join(seq_subdirectory, event.file_name)
            )
            self.write_to_logfile(
                f"Save sequences to {event.file_name} in {seq_subdirectory}"
            )
        elif isinstance(event, ev.TrackParameters):
            write_dict_to_json(
                event.parameters,
                os.path.join(
                    self.directory,
                    f"{event.name}.json",
                ),
            )
            self.write_to_logfile(f"Save model config json file to {self.directory}")

    def write_to_logfile(self, msg: str) -> None:
        with open(self.log_file_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {msg}\n")
