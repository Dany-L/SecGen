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
PREPARED_FOLDER_NAME = "prepared" # after ood separation
PROCESSED_FOLDER_NAME = "processed" # split into train/val/test as csv
RAW_FOLDER_NAME = "raw" # data as downloaded
IN_DISTRIBUTION_FOLDER_NAME = "id-test"
OUT_OF_DISTRIBUTION_FOLDER_NAME = "ood-test"
TRAIN_FOLDER_NAME = "train"
VALIDATION_FOLDER_NAME = "validation"
TEST_FOLDER_NAME = "test"

SHIP_DOWNLOAD_URL = {
    'darus': {
        'base_url': "https://darus.uni-stuttgart.de/",
        'doi': 'doi:10.18419/darus-2905'
    }
}
MSD_DOWNLOAD_URL = {
    'darus':{
        'base_url': "https://darus.uni-stuttgart.de/",
        'doi': 'doi:10.18419/DARUS-4768'
    }
}
HYST_DOWNLOAD_URL = {
    'test':{
        'url':"https://data.4tu.nl/file/7060f9bc-8289-411e-8d32-57bef2740d32/cd40469c-5064-4968-ae59-88cbb850264b", 
        'file_type': 'zip',
        }, 
    'train': {
        'url':'https://figshare.com/ndownloader/files/58117315',
        'file_type': 'mat'
        }
    }
AVAILABLE_DATASETS = {
    "ship": SHIP_DOWNLOAD_URL,
    "msd": MSD_DOWNLOAD_URL,
    "hyst": HYST_DOWNLOAD_URL
}

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


@dataclass
class SystemIdentificationResults:
    """Results from N4SID system identification."""
    ss: Any  # State space model (scipy object)
    transient_time: int
    fit: Optional[float] = None
    rmse: Optional[NDArray[np.float64]] = None
    N: Optional[int] = None


@dataclass
class PreprocessingResults:
    """Complete preprocessing results containing all calculated parameters."""
    system_identification: SystemIdentificationResults
    horizon: int
    window: int
    normalization: Normalization
    dataset_info: Optional[Dict[str, Any]] = None


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
