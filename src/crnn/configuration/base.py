import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray

CONFIG_FILE_ENV_VAR = "CONFIGURATION"
DATASET_DIR_ENV_VAR = "DATASET_DIRECTORY"
RESULT_DIR_ENV_VAR = "RESULT_DIRECTORY"
FIG_FOLDER_NAME = "fig"
SEQ_FOLDER_NAME = "seq"
NORMALIZATION_FILENAME = "normalization"
INITIALIZATION_FILENAME = "initialization"
PROCESSED_FOLDER_NAME = "processed"
IN_DISTRIBUTION_FOLDER_NAME = "in-distribution"
OUT_OF_DISTRIBUTION_FOLDER_NAME = "out-of-distribution"


@dataclass
class NormalizationParameters:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


@dataclass
class InputOutput:
    d: NDArray[np.float64]
    e_hat: NDArray[np.float64]
    e: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.array([0.0]))
    x0: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.array([0.0]))

@dataclass
class InputOutputList:
    d: List[NDArray[np.float64]]
    e: List[NDArray[np.float64]]

    def __iter__(self):
        return iter((self.d, self.e))


@dataclass
class InitializationData:
    msg: str
    data: Dict[str, Any]


@dataclass
class Normalization:
    input: NormalizationParameters
    output: NormalizationParameters


def parse_input(parser: argparse.ArgumentParser) -> Tuple[str, str, bool]:
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("model", type=str, help="Name of the model to train")
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Train on cuda if it is available",
    )
    args = parser.parse_args()
    return (args.model, args.experiment_name, args.gpu)
