from typing import Dict, List, Literal

from ..configuration.experiment import BaseTrackerConfig
from ..tracker.base import BaseTracker, retrieve_tracker_class


def get_trackers_from_config(
    config: Dict[str, BaseTrackerConfig],
    result_directory: str,
    model_name: str,
    type: Literal["preprocessing", "training", "validation"],
) -> List[BaseTracker]:
    trackers = []
    for tracker_config in config.values():
        tracker_class = retrieve_tracker_class(tracker_config.tracker_class)
        trackers.append(
            tracker_class(tracker_config, result_directory, model_name, type)
        )
    return trackers
