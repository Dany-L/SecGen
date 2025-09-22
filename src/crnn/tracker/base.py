import dataclasses
import os
from abc import abstractmethod
from typing import List, Literal, Optional, Type

from pydantic import BaseModel


@dataclasses.dataclass
class Event:
    msg: str


class TrackerConfig(BaseModel):
    id: Optional[str] = ""


class BaseTracker:
    CONFIG = TrackerConfig

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        directory: str = os.environ["HOME"],
        model_name: str = "",
        type: Literal["preprocessing", "training", "validation"] = "training",
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


_aggregated_tracker_instance = None


def set_aggregated_tracker(tracker: AggregatedTracker) -> None:
    global _aggregated_tracker_instance
    _aggregated_tracker_instance = tracker


def get_aggregated_tracker() -> AggregatedTracker:
    return _aggregated_tracker_instance


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
