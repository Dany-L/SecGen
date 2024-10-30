from abc import abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray


class Metrics:
    def __init__(self):
        pass

    @abstractmethod
    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        pass


class Rmse(Metrics):
    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        M = len(es)
        h, ne = es[0].shape
        rmse = np.zeros((ne))
        for e, e_hat in zip(es, e_hats):
            for e_idx in range(ne):
                rmse[e_idx] += 1 / h * np.sum((e[:, e_idx] - e_hat[:, e_idx]) ** 2)
        return np.sqrt(1 / M * rmse)
