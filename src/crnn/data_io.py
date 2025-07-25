import json
import os
from typing import Any, Dict, List, Literal, Tuple, Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

from .configuration.base import (
    DATASET_DIR_ENV_VAR,
    INITIALIZATION_FILENAME,
    NORMALIZATION_FILENAME,
    PROCESSED_FOLDER_NAME,
    InitializationData,
    InputOutput,
    Normalization,
    NormalizationParameters,
    InputOutputList,
)
from .models import base as base
from .models.jax import base as base_jax
from .models.jax.recurrent import get_matrices_from_flat_theta
from .models.torch import base as base_torch

import nonlinear_benchmarks


def load_data(
    input_names: List[str],
    output_names: List[str],
    type: Literal["train", "test", "validation"],
    dataset_dir: str,
) -> InputOutputList:
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

    type_path: str = os.path.join(dataset_dir, type)
    if not os.path.exists(type_path):
        # check if dataset is available in nonlinear_benchmarks
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))
        if not hasattr(nonlinear_benchmarks, dataset_name):
            raise ValueError(
                f"Dataset files do not exist and is not part of the nonlinear benchmarks {dataset_dir} {dataset_name}"
            )
        else:
            # We use the convention that the dataset directory must match an existing nonlinear benchmark name
            # as of Feb 2025 the current names are supported
            # CED, Cascaded_Tanks, EMPS, Silverbox, WienerHammerBenchMark, ParWH, F16
            cls = getattr(nonlinear_benchmarks, dataset_name)
            return load_data_from_nonlinear_benchmarks(type_path, cls)

    else:
        return load_data_from_folder(type_path, input_names, output_names)


def load_data_from_nonlinear_benchmarks(
    type_dir: str, dataset: Callable, val_split: float = 0.1
) -> InputOutputList:
    dataset_dir = os.path.dirname(type_dir)
    train_val, test = dataset(atleast_2d=True, always_return_tuples_of_datasets=True)
    datasetname = dataset.__name__

    N_train_val, ne = train_val[0].y.shape
    _, nd = train_val[0].u.shape
    M_train_val, M_test = len(train_val), len(test)

    input_names = [f"u_{idx+1}" for idx in range(nd)]
    output_names = [f"y_{idx+1}" for idx in range(ne)]

    for m in range(M_train_val):
        write_train_val_data_to_csv_file(
            train_val[m],
            val_split,
            m,
            datasetname,
            dataset_dir,
            input_names,
            output_names,
            nd,
            ne,
        )

    for m in range(M_test):
        write_test_data_to_csv_file(
            test[m], m, datasetname, dataset_dir, input_names, output_names, nd, ne
        )

    return load_data_from_folder(type_dir, input_names, output_names)


def write_test_data_to_csv_file(
    data: nonlinear_benchmarks.Input_output_data,
    idx: int,
    datasetname: str,
    dataset_dir: str,
    input_names: List[str],
    output_names: List[str],
    nd: int,
    ne: int,
) -> None:
    N_test = len(data)
    data.u = data.u.reshape(N_test, nd)
    data.y = data.y.reshape(N_test, ne)

    test_data = pd.DataFrame(
        np.hstack([data.u, data.y]), columns=input_names + output_names
    )

    directory = os.path.join(dataset_dir, "test")
    os.makedirs(directory, exist_ok=True)
    full_filename = os.path.join(directory, f"{datasetname}-{idx}.csv")
    test_data.to_csv(full_filename)


def write_train_val_data_to_csv_file(
    data: nonlinear_benchmarks.Input_output_data,
    val_split: float,
    idx: int,
    datasetname: str,
    dataset_dir: str,
    input_names: List[str],
    output_names: List[str],
    nd: int,
    ne: int,
) -> None:
    N_train_val = len(data)
    N_train = int(N_train_val * (1 - val_split))

    data.u = data.u.reshape(N_train_val, nd)
    data.y = data.y.reshape(N_train_val, ne)

    train_data = pd.DataFrame(
        np.hstack([data.u[:N_train], data.y[:N_train]]),
        columns=input_names + output_names,
    )
    val_data = pd.DataFrame(
        np.hstack([data.u[N_train:], data.y[N_train:]]),
        columns=input_names + output_names,
    )

    for data, t in zip([train_data, val_data], ["train", "validation"]):
        directory = os.path.join(dataset_dir, t)
        os.makedirs(directory, exist_ok=True)
        full_filename = os.path.join(directory, f"{datasetname}-{idx}.csv")
        data.to_csv(full_filename)


def load_data_from_folder(
    data_folder: str, input_names: List[str], output_names: List[str]
) -> InputOutputList:
    all_files: List[str] = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(".csv")
    ]
    if not all_files:
        raise ValueError(f"No CSV files found in type {type}.")

    inputs: List[NDArray[np.float64]] = []
    outputs: List[NDArray[np.float64]] = []

    for file in all_files:
        input, output = load_file(file, input_names, output_names)
        inputs.append(input)
        outputs.append(output)

    return InputOutputList(inputs, outputs)


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
                    dt=initialization["ss"]["dt"],
                ),
                "transient_time": initialization["transient_time"],
            },
        )
    else:
        return InitializationData("", {})


def split_trajectories(
    trajectories: List[np.ndarray], segment_length: int
) -> List[np.ndarray]:
    """Split each trajectory into non-overlapping subtrajectories of given length."""
    segments = []
    for traj in trajectories:
        num_segments = len(traj) // segment_length
        for i in range(num_segments):
            segment = traj[i * segment_length:(i + 1) * segment_length]
            segments.append(segment)
    return segments

def compute_energy(
    u_segment: np.ndarray
) -> float:
    """Compute energy of a signal segment as sum of squared values."""
    return float(np.sum(u_segment ** 2))

def process_data(
    u: List[np.ndarray],
    y: List[np.ndarray],
    w: int,
    h: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes trajectory data by splitting, computing energy, and separating high energy sequences.
    
    Returns:
        Tuple containing d_ood, training_set, validation_set
    """
    segment_length = w + h +1

    u_segments = split_trajectories(u, segment_length)
    y_segments = split_trajectories(y, segment_length)

    # Compute energy for each segment pair
    input_energies = [
        compute_energy(u_seg)
        for u_seg in u_segments
    ]

    # Sort indices by energy in descending order
    sorted_indices = np.argsort(input_energies)[::-1]

    # Top 10% high-energy samples
    num_top = max(1, int(0.1 * len(input_energies)))
    top_indices = sorted_indices[:num_top]
    remaining_indices = sorted_indices[num_top:]

    ood_input = [u_segments[i] for i in top_indices]
    ood_output = [y_segments[i] for i in top_indices]
    D_ood = (ood_input, ood_output)

    remaining_input = [u_segments[i] for i in remaining_indices]
    remaining_output =  [y_segments[i] for i in remaining_indices]

    # Shuffle and split remaining data
    d_train, d_val, e_train, e_val = train_test_split(
        remaining_input, remaining_output,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    D_train = (d_train, e_train)
    D_validation = (d_val, e_val)

    return D_ood, D_train, D_validation

def save_trajectories_to_csv(
    ood_input: List[np.ndarray],
    ood_output: List[np.ndarray],
    input_columns: List[str],
    output_columns: List[str],
    output_dir: str
) -> None:
    """
    Save each trajectory as a single CSV file with input and output signals combined.

    Args:
        ood_input: List of input trajectories (shape: [timesteps, input_dim]).
        ood_output: List of output trajectories (shape: [timesteps, output_dim]).
        input_columns: Column names for input signals.
        output_columns: Column names for output signals.
        output_dir: Directory to save the combined CSV files.
    """
    if os.path.exists(output_dir):
        print('ood already exists.')
        return
    os.makedirs(output_dir, exist_ok=True)
    num_trajectories = len(ood_input)
    assert num_trajectories == len(ood_output), "Input and output lists must be the same length."

    for i, (u_traj, y_traj) in enumerate(zip(ood_input, ood_output)):
        if u_traj.shape[0] != y_traj.shape[0]:
            raise ValueError(f"Trajectory {i} has mismatched time steps: input {u_traj.shape[0]} vs output {y_traj.shape[0]}")
        if u_traj.shape[1] != len(input_columns):
            raise ValueError(f"Input trajectory {i} has shape {u_traj.shape}, expected {len(input_columns)} input columns.")
        if y_traj.shape[1] != len(output_columns):
            raise ValueError(f"Output trajectory {i} has shape {y_traj.shape}, expected {len(output_columns)} output columns.")

        # Concatenate input and output signals
        combined = np.concatenate([u_traj, y_traj], axis=1)
        combined_columns = input_columns + output_columns

        df = pd.DataFrame(combined, columns=combined_columns)

        filepath = os.path.join(output_dir, f"traj_{i:03d}.csv")
        df.to_csv(filepath, index=False)

    print(f"✅ Saved {num_trajectories} combined input/output trajectory files to '{output_dir}'")