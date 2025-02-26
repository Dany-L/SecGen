import json
import os
from typing import Any, Dict, List, Literal, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.io import loadmat, savemat

from .configuration.base import (DATASET_DIR_ENV_VAR, INITIALIZATION_FILENAME,
                                 NORMALIZATION_FILENAME, InitializationData,
                                 InputOutput, Normalization,
                                 NormalizationParameters)
from .models import base as base
from .models.jax import base as base_jax
from .models.jax.recurrent import get_matrices_from_flat_theta
from .models.torch import base as base_torch


def load_data(
    input_names: List[str],
    output_names: List[str],
    type: Literal["train", "test", "validation"],
    dataset_dir: str,
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


def get_result_directory_name(
    directory: str, model_name: str, experiment_name: str
) -> str:
    result_directory = os.path.join(directory, f"{experiment_name}-{model_name}")
    os.makedirs(result_directory, exist_ok=True)
    return result_directory


def save_model(
    model: base.DynamicIdentificationModel, directory_name: str, file_name: str
) -> None:
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
    if isinstance(model, base_jax.ConstrainedModule):
        par_dict = {
            n: p
            for n, p in zip(
                model.parameter_names, get_matrices_from_flat_theta(model, model.theta)
            )
        }
        jnp.savez(file_path, **par_dict)
    elif isinstance(model, base_torch.DynamicIdentificationModel):
        torch.save(model.state_dict(), f"{file_path}.pth")


def save_model_parameter(
    model: base.DynamicIdentificationModel, file_name: str
) -> None:
    if isinstance(model, base_jax.ConstrainedModule):
        par_dict = {n: np.array(p) for n, p in zip(model.parameter_names, model.theta)}
        savemat(f"{file_name}.mat", par_dict)
    elif isinstance(model, base_torch.DynamicIdentificationModel):
        par_dict = {}
        trained_parameters = []
        for name, var in model.named_parameters():
            par_dict[name] = var.cpu().detach().numpy()
            if var.requires_grad:
                trained_parameters.append(name)
        parameter_json_file_name = f"{file_name}.json"
        if os.path.exists(parameter_json_file_name):
            with open(parameter_json_file_name, mode="r") as f:
                par_dict_from_file = json.load(f)
            new_trained_parameters = set(
                par_dict_from_file["trained_parameters"]
            ) - set(trained_parameters)
            trained_parameters.extend(new_trained_parameters)
        write_dict_to_json(
            {
                "trained_parameters": trained_parameters,
                "all_parameters": list(par_dict.keys()),
            },
            parameter_json_file_name,
        )
        savemat(f"{file_name}.mat", par_dict)


def save_input_output_to_mat(sequences: List[InputOutput], file_name) -> None:
    e_hats = [s.e_hat for s in sequences]
    es = [s.e for s in sequences]
    ds = [s.d for s in sequences]
    x0s = [s.x0 for s in sequences]
    savemat(
        f"{file_name}.mat",
        {
            "e_hat": np.array(e_hats),
            "e": np.array(es),
            "d": np.array(ds),
            "x0": np.array(x0s),
        },
    )


def load_input_output_from_mat(filename: str) -> List[InputOutput]:
    input_output_data = loadmat(filename)
    N = input_output_data["e_hat"].shape[0]
    names = InputOutput.__annotations__.keys()
    ios = []
    for N_idx in range(N):
        io = {}
        for name in names:
            # if len(input_output_data[name].shape) > 2:
            io[name] = input_output_data[name][N_idx]
        ios.append(InputOutput(**io))
    return ios


def write_dict_to_json(config: Dict[str, Any], file_name: str) -> None:
    with open(file_name, "w") as f:
        json.dump(config, f)


def load_normalization(directory: str) -> Normalization:
    with open(
        os.path.join(directory, f"{NORMALIZATION_FILENAME}.json"),
        "r",
    ) as f:
        normalization = json.load(f)
    return Normalization(
        input=NormalizationParameters(
            mean=normalization["input_mean"], std=normalization["input_std"]
        ),
        output=NormalizationParameters(
            mean=normalization["output_mean"], std=normalization["output_std"]
        ),
    )


def load_initialization(directory: str) -> InitializationData:
    filename = os.path.join(directory, f"{INITIALIZATION_FILENAME}.json")
    if os.path.exists(filename):
        with open(
            filename,
            "r",
        ) as f:
            initialization = json.load(f)
        return InitializationData(
            "",
            {
                "ss": base.Linear(
                    torch.tensor(initialization["ss"]["A"]),
                    torch.tensor(initialization["ss"]["B"]),
                    torch.tensor(initialization["ss"]["C"]),
                    torch.tensor(initialization["ss"]["D"]),
                )
            },
        )
    else:
        return InitializationData("", {})
