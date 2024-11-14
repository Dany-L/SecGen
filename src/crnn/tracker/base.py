import dataclasses
from abc import abstractmethod
from typing import List


@dataclasses.dataclass
class Event:
    msg: str


class BaseTracker:
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def track(self, event: Event) -> None:
        pass


class AggregatedTracker:
    def __init__(self, trackers: List[BaseTracker] = None) -> None:
        self.trackers = trackers

    def track(self, event: Event) -> None:
        for tracker in self.trackers:
            tracker.track(event)
