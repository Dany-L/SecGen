import os
import pandas as pd
from typing import Optional, Literal, List, Tuple
from numpy.typing import NDArray
import numpy as np

DATASET_DIR_ENV_VAR = 'DATASET_DIRECTORY'

def load_data(
    input_names: List[str],
    output_names: List[str],
    type: Literal['train', 'test', 'validation']
    ) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
    """
    Load data from CSV files in the specified type directory ('train', 'test', 'validation').

    Args:
    input_names (List[str]): List of input column names to load.
    output_names (List[str]): List of output column names to load.
    type (Literal['train', 'test', 'validation']): The type of data to load.

    Returns:
    Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]: A tuple containing two lists of NumPy arrays,
    one for the input columns and one for the output columns, from all CSV files in the specified type directory.
    """
    dataset_dir: Optional[str] = os.getenv(os.path.expanduser(DATASET_DIR_ENV_VAR))
    if not dataset_dir:
        raise ValueError(f"Environment variable {DATASET_DIR_ENV_VAR} is not set.")
    
    type_path: str = os.path.join(dataset_dir, type)
    if not os.path.exists(type_path):
        raise ValueError(f"type {type} does not exist in {dataset_dir}.")
    
    all_files: List[str] = [os.path.join(type_path, f) for f in os.listdir(type_path) if f.endswith('.csv')]
    if not all_files:
        raise ValueError(f"No CSV files found in type {type}.")

    inputs, outputs = zip(*[load_file(file, input_names, output_names) for file in all_files])
    
    return (inputs, outputs)

def load_file(file_path: str, input_names: List[str], output_names: List[str]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load specific columns from a CSV file and return them as NumPy arrays.

    Args:
        file_path (str): The path to the CSV file.
        input_names (List[str]): List of input column names to load.
        output_names (List[str]): List of output column names to load.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays, one for the input columns and one for the output columns.
    """
    columns_to_load = input_names + output_names
    df = pd.read_csv(file_path, usecols=columns_to_load)
    input_data = df[input_names].to_numpy()
    output_data = df[output_names].to_numpy()
    return input_data, output_data
