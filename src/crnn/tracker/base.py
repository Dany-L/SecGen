import dataclasses
import os
from abc import abstractmethod
from typing import Dict, List, Literal, Optional, Type

from pydantic import BaseModel


@dataclasses.dataclass
class Event:
    msg: str


class BaseTrackerConfig(BaseModel):
    id: Optional[str] = ""


class BaseTracker:
    CONFIG = BaseTrackerConfig

    def __init__(
        self,
        config: Optional[BaseTrackerConfig] = None,
        directory: str = os.environ["HOME"],
        model_name: str = "",
        type: Literal["training", "validation"] = "training",
    ) -> None:
        self.directory = directory
        self.model_name = model_name
        self.log_file_path = str(os.path.join(self.directory, f"{type}.log"))

    @abstractmethod
    def track(self, event: Event) -> None:
        pass


class AggregatedTracker:
    def __init__(self, trackers: List[BaseTracker] = []) -> None:
        self.trackers = trackers

    def track(self, event: Event) -> None:
        for tracker in self.trackers:
            tracker.track(event)


def get_trackers_from_config(
    config: Dict[str, BaseTrackerConfig],
    result_directory: str,
    model_name: str,
    type: Literal["training", "validation"],
) -> List[BaseTracker]:
    trackers = []
    for tracker_config in config.values():
        tracker_class = retrieve_tracker_class(tracker_config.tracker_class)
        trackers.append(
            tracker_class(tracker_config, result_directory, model_name, type)
        )
    return trackers


def retrieve_tracker_class(tracker_class_string: str) -> Type[BaseTracker]:
    # https://stackoverflow.com/a/452981
    parts = tracker_class_string.split(".")
    module_string = ".".join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, BaseTracker):
        raise ValueError(f"{cls} is not a subclass of BaseTracker")
    return cls  # type: ignore
