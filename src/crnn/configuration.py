import json
import os
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel

CONFIG_FOLDER_ENV_VAR = "CONFIG_DIRECTORY"
DATASET_DIR_ENV_VAR = "DATASET_DIRECTORY"
RESULT_DIR_ENV_VAR = "RESULT_DIRECTORY"
FIG_FOLDER_NAME = "fig"


@dataclass
class LureSystemClass:
    A: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    D12: torch.Tensor
    C2: torch.Tensor
    D21: torch.Tensor
    D22: torch.Tensor
    Delta: torch.nn.Module


@dataclass
class NormalizationParameters:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


@dataclass
class Normalization:
    input: NormalizationParameters
    output: NormalizationParameters


class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd"]
    learning_rate: float
    initial_hidden_state: Literal["zero", "joint", "separate"]


class HorizonsConfig(BaseModel):
    training: int
    testing: int


class BaseConfig(BaseModel):
    epochs: int
    eps: float
    optimizer: OptimizerConfig
    num_layer: int
    dt: float
    nz: int
    batch_size: int
    window: int
    loss_function: Literal["mse"]
    input_names: List[str]
    output_names: List[str]
    horizons: HorizonsConfig


def load_configuration() -> BaseConfig:
    config_dir = os.path.expanduser(os.getenv(CONFIG_FOLDER_ENV_VAR))
    base_config_file_name = os.path.join(config_dir, "base.json")
    with open(base_config_file_name, "r") as file:
        config_data = json.load(file)
    return BaseConfig(**config_data)
