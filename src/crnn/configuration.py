import json
import os
from dataclasses import dataclass
from typing import List, Literal

import torch
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


class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd"]
    learning_rate: float


class HorizonsConfig(BaseModel):
    training: int
    testing: int


class BaseConfig(BaseModel):
    epochs: int
    eps: float
    optimizer: OptimizerConfig
    dt: float
    nz: int
    batch_size: int
    window: int
    loss_function: Literal["mse"]
    input_names: List[str]
    output_names: List[str]
    horizons: HorizonsConfig


def load_configuration() -> BaseConfig:
    config_dir = os.getenv(os.path.expanduser(CONFIG_FOLDER_ENV_VAR))
    base_config_file_name = os.path.join(config_dir, "base.json")
    with open(base_config_file_name, "r") as file:
        config_data = json.load(file)
    return BaseConfig(**config_data)
