from pydantic import BaseModel
import torch
import os
import json
from typing import List, Callable, Literal
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

CONFIG_FOLDER_ENV_VAR = 'CONFIG_DIRECTORY'
DATASET_DIR_ENV_VAR = 'DATASET_DIRECTORY'
RESULT_DIR_ENV_VAR = 'RESULT_DIRECTORY'

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
    name: Literal['adam', 'sgd']
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
    loss_function: Literal['mse']
    input_names: List[str]
    output_names: List[str]
    horizons: HorizonsConfig

def load_configuration()-> BaseConfig:
    config_dir: str = os.getenv(os.path.expanduser(CONFIG_FOLDER_ENV_VAR))
    base_config_filename = os.path.join(config_dir, 'base.json')
    with open(base_config_filename, 'r') as file:
        config_data = json.load(file)
    return BaseConfig(**config_data)


