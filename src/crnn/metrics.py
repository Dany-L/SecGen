from abc import abstractmethod
from typing import List, Type, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class MetricConfig(BaseModel):
    pass


class Metrics:
    CONFIG: Type[MetricConfig] = MetricConfig

    def __init__(self, config: MetricConfig):
        pass

    @abstractmethod
    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        pass


class Rmse(Metrics):
    def __init__(self, config: MetricConfig):
        pass

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
    
class Nrmse(Metrics):
    def __init__(self, config: MetricConfig):
        pass

    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        M = len(es)
        std = np.std(np.vstack(es), axis=0)
        h, ne = es[0].shape
        nrmse = np.zeros((ne))
        for e, e_hat in zip(es, e_hats):
            for e_idx in range(ne):
                nrmse[e_idx] += 1/h * np.sum((e[:, e_idx] - e_hat[:, e_idx]) ** 2) 
        return 1/std * np.sqrt(1 / M * nrmse)
    
class Fit(Metrics):
    def __init__(self, config: MetricConfig):
        pass

    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        M = len(es)
        h, ne = es[0].shape
        mean = np.mean(np.vstack(es), axis=0)
        fit = np.zeros((ne))
        for e, e_hat in zip(es, e_hats):
            for e_idx in range(ne):
                fit[e_idx] += 1 - np.sum((e[:, e_idx] - e_hat[:, e_idx])**2) / np.sum((e[:, e_idx] - mean[e_idx])**2)
        return 100 * fit / M
    
class PsdRmse(Metrics):
    def __init__(self, config: MetricConfig):
        pass

    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        M = len(es)
        ne = es[0].shape[1]
        psd_rmse = np.zeros((ne))
        for e, e_hat in zip(es, e_hats):
            for e_idx in range(ne):
                f_e, P_e = self._compute_psd(e[:, e_idx])
                f_e_hat, P_e_hat = self._compute_psd(e_hat[:, e_idx])
                psd_rmse[e_idx] += np.mean((P_e - P_e_hat) ** 2)
        return np.sqrt(1 / M * psd_rmse)

    def _compute_psd(self, signal: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        from scipy.signal import welch

        fs = 1.0  # Sampling frequency is 1 Hz (arbitrary units)
        f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
        return f, Pxx
    
class EnergyMse(Metrics):
    def __init__(self, config: MetricConfig):
        pass

    def forward(
        self, es: List[NDArray[np.float64]], e_hats: List[NDArray[np.float64]]
    ) -> np.float64:
        def E_k(signal: NDArray[np.float64]) -> float:
            return signal.reshape(-1,1).T @ signal.reshape(-1,1)
        M = len(es)
        N = es[0].shape[0]
        Ey, E_hat_y = 0.0, 0.0
        energy_mse = 0.0
        for e, e_hat in zip(es, e_hats):
            for k in range(N):
                Ey += E_k(e[k, :])
                E_hat_y += E_k(e_hat[k, :]) 
            energy_mse+=np.abs(Ey-E_hat_y)/Ey

        return np.squeeze(1/M * energy_mse)


def retrieve_metric_class(metric_class_string: str) -> Type[Metrics]:
    # https://stackoverflow.com/a/452981
    parts = metric_class_string.split(".")
    module_string = ".".join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, Metrics):
        raise ValueError(f"{cls} is not a subclass of Metrics.")
    return cls  # type: ignore
