from typing import List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
from numpy.typing import NDArray


def get_duration_str(start_time: float, end_time: float) -> str:
    duration = end_time - start_time
    days, remainder = divmod(duration, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(days):02}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def normalize(
    data: List[NDArray[np.float64]], mean: NDArray[np.float64], std: NDArray[np.float64]
) -> List[NDArray[np.float64]]:
    return [(d - mean) / std for d in data]


def denormalize(
    data: List[NDArray[np.float64]], mean: NDArray[np.float64], std: NDArray[np.float64]
) -> List[NDArray[np.float64]]:
    return [d * std + mean for d in data]


def get_mean_std(
    data: List[NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.concatenate(data, axis=0)
    return np.mean(data, axis=0), np.std(data, axis=0)


def get_model_file_name(name: str, model_name: str) -> str:
    return f"parameters-{name}-{model_name}.pth"


def get_config_file_name(name: str, model_name: str) -> str:
    return f"config-{name}-{model_name}"


def get_opt_values(
    opt_vars: Union[NDArray[np.float64], cp.Expression]
) -> NDArray[np.float64]:
    if isinstance(opt_vars, cp.Expression):
        return opt_vars.value
    return opt_vars


def get_device(gpu: bool) -> torch.device:
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
