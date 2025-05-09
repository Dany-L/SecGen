import json
import os
from datetime import datetime

from ..configuration.base import (
    FIG_FOLDER_NAME,
    INITIALIZATION_FILENAME,
    NORMALIZATION_FILENAME,
    SEQ_FOLDER_NAME,
)
from ..data_io import (
    save_input_output_to_mat,
    save_model,
    save_model_parameter,
    write_dict_to_json,
)
from ..utils import base as utils
from ..utils import plot
from . import events as ev
from .base import BaseTracker, Event, TrackerConfig


class IoTrackerConfig(TrackerConfig):
    pass


class IoTracker(BaseTracker):
    CONFIG = IoTrackerConfig

    def track(self, event: Event) -> None:
        if isinstance(event, ev.Log):
            print(event.log_msg)
            self.write_to_logfile(event.log_msg)
        elif isinstance(event, ev.SaveNormalization):
            with open(
                os.path.join(self.directory, f"{NORMALIZATION_FILENAME}.json"),
                "w",
            ) as f:
                json.dump(
                    dict(
                        input_mean=event.input.mean.tolist(),
                        input_std=event.input.std.tolist(),
                        output_mean=event.output.mean.tolist(),
                        output_std=event.output.std.tolist(),
                    ),
                    f,
                )
            self.write_to_logfile(
                f"Save normalization mean and std to {self.directory}"
            )
        elif isinstance(event, ev.SaveInitialization):
            ss = dict(
                A=event.ss.A.cpu().detach().numpy().tolist(),
                B=event.ss.B.cpu().detach().numpy().tolist(),
                C=event.ss.C.cpu().detach().numpy().tolist(),
                D=event.ss.D.cpu().detach().numpy().tolist(),
                dt=event.ss.dt,
            )
            event.data["ss"] = ss
            with open(
                os.path.join(self.directory, f"{INITIALIZATION_FILENAME}.json"),
                "w",
            ) as f:
                json.dump(event.data, f)
            self.write_to_logfile(f"Save initialization to {self.directory}")
        elif isinstance(event, ev.SaveFig):
            fig_subdirectory = os.path.join(self.directory, FIG_FOLDER_NAME)
            os.makedirs(fig_subdirectory, exist_ok=True)
            plot.save_fig(event.fig, event.name, fig_subdirectory)
            self.write_to_logfile(f"save fig {event.name} in {self.directory}")
        elif isinstance(event, ev.Start):
            self.write_to_logfile(f"--- Start model {self.model_name} ---")
        elif isinstance(event, ev.Stop):
            self.write_to_logfile("--- Stop ---")
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
                os.path.join(
                    self.directory, f"model_params-{self.model_name}{event.name_suffix}"
                ),
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
        elif isinstance(event, ev.TrackResults):
            write_dict_to_json(
                event.parameters,
                os.path.join(
                    self.directory,
                    f"{event.name}.json",
                ),
            )
            self.write_to_logfile(f"Save results json to {self.directory}")

    def write_to_logfile(self, msg: str) -> None:
        with open(self.log_file_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {msg}\n")
