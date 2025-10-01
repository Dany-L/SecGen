import json
import os
from typing import Literal, Optional

import mlflow

from ..configuration.base import FIG_FOLDER_NAME
from . import events as ev
from .base import BaseTracker, TrackerConfig


class MlflowConfig(TrackerConfig):
    tracking_uri: Optional[str] = None


class MlFlowTracker(BaseTracker):
    CONFIG = MlflowConfig

    def __init__(
        self,
        config: MlflowConfig,
        directory: str = os.environ["HOME"],
        model_name: str = "",
        type: Literal["preprocessing", "training", "validation"] = "training",
        dataset_directory: str = os.environ["HOME"]
    ) -> None:
        super().__init__(config, directory, model_name, type, dataset_directory)
        if hasattr(config.parameters, "tracking_uri"):
            # if config.tracking_uri is not None:
            mlflow.set_tracking_uri(config.parameters.tracking_uri)

    def track(self, event: ev.Event) -> None:
        if isinstance(event, ev.TrackParameters):
            for key, value in event.parameters.items():
                mlflow.log_param(key, value)
        if isinstance(event, ev.TrackParameter):
            mlflow.log_param(event.name, event.parameter)
        elif isinstance(event, ev.Start):
            # experiment_name = os.path.split(pathlib.Path(self.directory).parent.parent)[
            #     1
            # ]
            mlflow.set_experiment(event.dataset_name)
        elif isinstance(event, ev.SaveTrackingConfiguration):
            self.save_tracking_configuration(event)
        elif isinstance(event, ev.Stop):
            mlflow.end_run()
        elif isinstance(event, ev.SetTags):
            for key, value in event.tags.items():
                mlflow.set_tag(key, value)
        elif isinstance(event, ev.LoadTrackingConfiguration):
            self.load_tracking_configuration(event)
        elif isinstance(event, ev.TrackMetrics):
            for key, value in event.metrics.items():
                mlflow.log_metric(key, value, event.step)
        elif isinstance(event, ev.SaveFig):
            mlflow.log_figure(event.fig, f"{FIG_FOLDER_NAME}/{event.name}.png")

    def save_tracking_configuration(self, event: ev.SaveTrackingConfiguration) -> None:
        run = mlflow.active_run()
        tracker_config_file_name = os.path.join(
            event.model_directory,
            build_tracker_config_file_name(event.model_name),
        )
        # if run is not None and not(os.path.exists(tracker_config_file_name)):
        for tracker_config in event.config.values():
            if (
                tracker_config.tracker_class
                == f"{self.__module__}.{self.__class__.__name__}"
            ):
                tracker_config.parameters.id = run.info.run_id
                with open(
                    tracker_config_file_name,
                    mode="w",
                ) as f:
                    f.write(tracker_config.parameters.model_dump_json())

    def load_tracking_configuration(self, event: ev.LoadTrackingConfiguration) -> None:
        config_file = os.path.join(
            event.model_directory, build_tracker_config_file_name(event.model_name)
        )
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            tracker_config = self.CONFIG(**config)
            mlflow.start_run(run_id=tracker_config.id)


def build_tracker_config_file_name(model_name: str) -> str:
    return f"tracker-{model_name}.json"
