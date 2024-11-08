from abc import abstractmethod
from typing import List, Type, Dict
from pydantic import BaseModel

import numpy as np
from numpy.typing import NDArray


class MetricConfig(BaseModel):
    pass

class Metrics:
    CONFIG: Type[MetricConfig] = MetricConfig
    def __init__(self, config: MetricConfig):
        pass

    @abstractmethod
    def forward(self,
        es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        pass


class Rmse(Metrics):
    def __init__(self, config: MetricConfig):
        pass

    def forward(self,
        es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        M = len(es)
        h, ne = es[0].shape
        rmse = np.zeros((ne))
        for e, e_hat in zip(es, e_hats):
            for e_idx in range(ne):
                rmse[e_idx] += 1 / h * np.sum((e[:, e_idx] - e_hat[:, e_idx]) ** 2)
        return np.sqrt(1 / M * rmse)


def retrieve_metric_class(metric_class_string: str) -> Type[Metrics]:
    # https://stackoverflow.com/a/452981
    parts = metric_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, Metrics):
        raise ValueError(f'{cls} is not a subclass of Metrics.')
    return cls  # type: ignore