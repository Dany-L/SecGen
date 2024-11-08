import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.io import savemat

from .configuration import (DATASET_DIR_ENV_VAR, RESULT_DIR_ENV_VAR, Normalization, NormalizationParameters,
                            ExperimentBaseConfig, DynamicIdentificationConfig)
from .models.base import ConstrainedModule


def load_data(
    input_names: List[str],
    output_names: List[str],
    type: Literal["train", "test", "validation"],
    dataset_dir: str
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
    """
    Load data from CSV files in the specified type directory ('train', 'test', 'validation').

    Args:
    input_names (List[str]): List of input column names to load.
    output_names (List[str]): List of output column names to load.
    type (Literal['train', 'test', 'validation']): The type of data to load.

    Returns:
    Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]: A Tuple containing two lists of NumPy arrays,
    one for the input columns and one for the output columns, from all CSV files in the specified type directory.
    """
    if not dataset_dir:
        raise ValueError(f"Environment variable {DATASET_DIR_ENV_VAR} is not set.")

    type_path: str = os.path.join(dataset_dir, type)
    if not os.path.exists(type_path):
        raise ValueError(f"type {type} does not exist in {dataset_dir}.")

    all_files: List[str] = [
        os.path.join(type_path, f) for f in os.listdir(type_path) if f.endswith(".csv")
    ]
    if not all_files:
        raise ValueError(f"No CSV files found in type {type}.")

    inputs: List[NDArray[np.float64]] = []
    outputs: List[NDArray[np.float64]] = []

    for file in all_files:
        input, output = load_file(file, input_names, output_names)
        inputs.append(input)
        outputs.append(output)

    return (inputs, outputs)


def load_file(
    file_path: str, input_names: List[str], output_names: List[str]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load specific columns from a CSV file and return them as NumPy arrays.

    Args:
        file_path (str): The path to the CSV file.
        input_names (List[str]): List of input column names to load.
        output_names (List[str]): List of output column names to load.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A Tuple containing two NumPy arrays, one for the input columns and one for the output columns.
    """
    columns_to_load = input_names + output_names
    df = pd.read_csv(file_path, usecols=columns_to_load)
    input_data = np.array(df[input_names].to_numpy())
    output_data = np.array(df[output_names].to_numpy())
    return (input_data, output_data)


def get_result_directory_name(directory:str, model_name: str, experiment_name:str) -> str:
    result_directory = os.path.join(directory, f'{experiment_name}-{model_name}')
    os.makedirs(result_directory, exist_ok=True)
    return result_directory


def save_model(model: ConstrainedModule, directory_name: str, file_name: str) -> None:
    """
    Save the parameters of a PyTorch model to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        directory_name (str): The name of the subdirectory to save the model in.
        file_name (str): The name of the file to save the model parameters to.

    Returns:
        str: The path to the saved model file.
    """
    file_path = os.path.join(directory_name, file_name)
    torch.save(model.state_dict(), file_path)


def save_model_parameter(model: ConstrainedModule, file_name: str) -> None:
    par_dict = {}
    for name, var in model.named_parameters():
        par_dict[name] = var.detach().numpy()
    savemat(file_name, par_dict)


def save_sequences_to_mat(
    e_hats: List[NDArray[np.float64]], es: List[NDArray[np.float64]], file_name
) -> None:
    savemat(f"{file_name}.mat", {"e_hat": np.array(e_hats), "e": np.array(es)})


def write_config(config: Union[ExperimentBaseConfig,DynamicIdentificationConfig], file_name: str) -> None:
    with open(file_name, "w") as f:
        f.write(config.model_dump_json())


def load_normalization(directory: str) -> Normalization:
    normalization = np.load(
        os.path.join(directory, "normalization.npz"), allow_pickle=True
    )
    return Normalization(
        input=NormalizationParameters(
            mean=normalization["input_mean"], std=normalization["input_std"]
        ),
        output=NormalizationParameters(
            mean=normalization["output_mean"], std=normalization["output_std"]
        ),
    )
