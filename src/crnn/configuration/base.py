import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

CONFIG_FILE_ENV_VAR = "CONFIGURATION"
DATASET_DIR_ENV_VAR = "DATASET_DIRECTORY"
RESULT_DIR_ENV_VAR = "RESULT_DIRECTORY"
FIG_FOLDER_NAME = "fig"


@dataclass
class NormalizationParameters:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


@dataclass
class InputOutput:
    d: NDArray[np.float64]
    e_hat: NDArray[np.float64]
    e: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.array([0.0]))


@dataclass
class Normalization:
    input: NormalizationParameters
    output: NormalizationParameters


def parse_input(parser: argparse.ArgumentParser) -> Tuple[str, str]:
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("model", type=str, help="Name of the model to train")
    args = parser.parse_args()
    return (args.model, args.experiment_name)