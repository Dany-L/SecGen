import json
import os
import io
import tarfile
import shutil
from typing import Any, Dict, List, Literal, Tuple, Callable, Union, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from ..tracker import events as ev
from ..tracker.base import get_aggregated_tracker
from ..utils import base as utils

from crnn.configuration.experiment import SplitConfig, BaseExperimentConfig

from ..configuration.base import (
    AVAILABLE_DATASETS,
    PREPARED_FOLDER_NAME,
    INITIALIZATION_FILENAME,
    NORMALIZATION_FILENAME,
    TRAIN_FOLDER_NAME,
    VALIDATION_FOLDER_NAME,
    TEST_FOLDER_NAME,
    IN_DISTRIBUTION_FOLDER_NAME,
    PROCESSED_FOLDER_NAME,
    RAW_FOLDER_NAME,
    OUT_OF_DISTRIBUTION_FOLDER_NAME,
    InitializationData,
    InputOutput,
    Normalization,
    NormalizationParameters,
    InputOutputList,
    PreprocessingResults,
    SystemIdentificationResults
)
from ..models import base as base

import nonlinear_benchmarks
import requests
import zipfile


def download_ship(base_url: str, doi: str, dataset_dir: str) -> None:
    tracker = get_aggregated_tracker()
    # Get dataset metadata
    api_url = f"{base_url.rstrip('/')}/api/datasets/:persistentId/?persistentId={doi}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    dataset_info = resp.json()
    file_list = dataset_info["data"]["latestVersion"]["files"]
    for file in file_list:
        file_name = file["dataFile"]["filename"]
        file_id = file["dataFile"]["id"]
        directory_label = file.get("directoryLabel", None)
        if directory_label is None:
            tracker.track(ev.Log("", f"Skipping {file_name}. Not a dataset file."))
            continue
        directory_root, *directory_elements = os.path.normpath(directory_label).split(
            os.sep
        )
        if "train" in directory_elements and directory_root == "patrol_ship_routine":
            directory = os.path.expanduser(
                os.path.join(dataset_dir, PROCESSED_FOLDER_NAME, TRAIN_FOLDER_NAME)
            )
        elif (
            "validation" in directory_elements
            and directory_root == "patrol_ship_routine"
        ):
            directory = os.path.expanduser(
                os.path.join(dataset_dir, PROCESSED_FOLDER_NAME, VALIDATION_FOLDER_NAME)
            )
        elif "test" in directory_elements and directory_root == "patrol_ship_routine":
            directory = os.path.expanduser(
                os.path.join(
                    dataset_dir, PROCESSED_FOLDER_NAME, IN_DISTRIBUTION_FOLDER_NAME
                )
            )
        elif "test" in directory_elements and directory_root == "patrol_ship_ood":
            directory = os.path.expanduser(
                os.path.join(
                    dataset_dir, PROCESSED_FOLDER_NAME, OUT_OF_DISTRIBUTION_FOLDER_NAME
                )
            )
        else:
            tracker.track(
                ev.Log(
                    "",
                    f"Unexpected directory {directory_root} encountered. "
                    'Does not match "patrol_ship_routine" or "patrol_ship_ood". '
                    "Skipping.",
                )
            )
            continue
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, os.path.splitext(file_name)[0] + ".csv")
        if os.path.exists(file_path):
            tracker.track(
                ev.Log("", f"File {file_path} already exists. Skipping download.")
            )
            continue

        tracker.track(ev.Log("", f"Downloading file to {file_path}."))
        # Download the file using the REST API
        file_url = f"{base_url.rstrip('/')}/api/access/datafile/{file_id}"
        file_resp = requests.get(file_url, stream=True)
        file_resp.raise_for_status()
        # Assume text/csv or text/tsv
        content_type = file_resp.headers.get("content-type", "")
        if "text/tab-separated-values" in content_type or file_name.endswith(".tsv"):
            sep = "\t"
        else:
            sep = ","
        # Read and save as CSV
        file_content = file_resp.content.decode("utf-8")
        df = pd.read_csv(io.StringIO(file_content), sep=sep)
        df.to_csv(file_path, index=False)


def download_msd(base_url: str, doi: str, dataset_dir: str) -> None:
    raw_dir = os.path.join(dataset_dir, RAW_FOLDER_NAME)

    tracker = get_aggregated_tracker()
    # Get dataset metadata to find the file ID for the zip file
    api_url = f"{base_url.rstrip('/')}/api/datasets/:persistentId/?persistentId={doi}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    dataset_info = resp.json()
    file_list = dataset_info["data"]["latestVersion"]["files"]
    # Find the zip file (assume only one zip file)
    zip_file = None
    for file in file_list:
        file_name = file["dataFile"]["filename"]
        if file_name.endswith(".zip"):
            zip_file = file
            break
    if zip_file is None:
        raise ValueError("No zip file found in the Dataverse dataset.")
    file_id = zip_file["dataFile"]["id"]
    # Download the zip file
    file_url = f"{base_url.rstrip('/')}/api/access/datafile/{file_id}"
    resp = requests.get(file_url, stream=True)
    resp.raise_for_status()
    # Save the zip file locally
    zip_path = os.path.join(raw_dir, zip_file["dataFile"]["filename"])
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    tracker.track(ev.Log("", f"✅ Downloaded and extracted dataset to {raw_dir}"))


def download_and_extract_dataset(
    urls: Dict[str, Dict[str, str]], dataset_name: str, dataset_dir: str
) -> None:
    """
    Download a dataset from a URL and extract it to the raw folder of the dataset directory.

    Args:
        url: URL to download the dataset from
        dataset_name: Name of the dataset (used for filename)
        dataset_dir: Directory where the dataset should be saved
    """
    tracker = get_aggregated_tracker()

    raw_dir = os.path.join(dataset_dir, RAW_FOLDER_NAME)
    if os.path.exists(raw_dir):
        print(f"📁 Raw directory '{raw_dir}' already exists. Skipping download.")
        return
    else:
        os.makedirs(raw_dir, exist_ok=True)

    for split, split_info in urls.items():

        if split == "darus":

            base_url = split_info.get("base_url")
            doi = split_info.get("doi")
            if not base_url or not doi:
                raise ValueError(
                    "Dataverse URL dictionary must contain 'base_url' and 'doi' keys."
                )
            download_function_name = f"download_{dataset_name}"
            if download_function_name in globals():
                download_function = globals()[download_function_name]
                try:
                    download_function(base_url, doi, dataset_dir)
                except Exception as e:
                    tracker.track(
                        ev.Log(
                            "",
                            f"❌ Error downloading dataset '{dataset_name}' from Dataverse: {e}",
                        )
                    )
            else:
                raise ValueError(
                    f"No download function found for dataset '{dataset_name}'. "
                    f"Please implement a function named '{download_function_name}' to handle this dataset."
                )

        elif split in ["train", "test", "validation"]:
            split_url = split_info["url"]
            file_type = split_info["file_type"]

            # Determine filename and whether it's an archive
            filename = f"{dataset_name}_{split}.{file_type}"
            is_archive = file_type in ["zip", "tar", "tar.gz"]

            filepath = os.path.join(raw_dir, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                print(f"📁 Dataset file '{filename}' already exists in '{raw_dir}'")
                continue

            try:
                print(
                    f"🔽 Downloading dataset '{dataset_name}' ({split}) from {split_url}"
                )

                # Download with progress indication
                response = requests.get(split_url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0

                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(
                                    f"\r📦 Download progress ({split}): {progress:.1f}%",
                                    end="",
                                    flush=True,
                                )

                print(f"\n✅ Downloaded '{filename}' to '{raw_dir}'")

                # Extract if it's an archive
                if is_archive:
                    print(f"📂 Extracting '{filename}'...")

                    if file_type == "zip":
                        with zipfile.ZipFile(filepath, "r") as zip_ref:
                            zip_ref.extractall(raw_dir)
                    elif file_type == "tar.gz":
                        with tarfile.open(filepath, "r:gz") as tar_ref:
                            tar_ref.extractall(raw_dir)
                    elif file_type == "tar":
                        with tarfile.open(filepath, "r") as tar_ref:
                            tar_ref.extractall(raw_dir)

                    print(f"✅ Extracted '{filename}' to '{raw_dir}'")

            except requests.exceptions.RequestException as e:
                print(f"❌ Error downloading dataset from {split_url}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise
            except zipfile.BadZipFile as e:
                print(f"❌ Error extracting zip file '{filename}': {e}")
                raise
            except Exception as e:
                print(
                    f"❌ Unexpected error processing dataset '{dataset_name}' ({split}): {e}"
                )
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise
        else:
            raise ValueError(
                f"Unknown split '{split}' in URL dictionary. Expected 'train', 'test', 'validation', or 'darus'."
            )


def check_data_availability(dataset_dir: str) -> Dict[str, bool]:
    """
    Check which data types are available in the dataset directory.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        Dict with availability status for each data type
    """
    availability = {}
    for data_type in [TRAIN_FOLDER_NAME, VALIDATION_FOLDER_NAME, IN_DISTRIBUTION_FOLDER_NAME, OUT_OF_DISTRIBUTION_FOLDER_NAME]:
        type_path = os.path.join(dataset_dir, PREPARED_FOLDER_NAME, data_type)
        availability[data_type] = (
            os.path.exists(type_path)
            and bool([f for f in os.listdir(type_path) if f.endswith(".csv")])
            if os.path.exists(type_path)
            else False
        )

    return availability


def download_and_prepare_data(dataset_dir: str, dt: float) -> bool:
    """
    Download and prepare dataset if it doesn't exist locally.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        bool: True if data was downloaded/prepared, False if already exists
    """
    # Extract dataset name from path structure
    dataset_name = utils.get_dataset_name(dataset_dir)

    if hasattr(nonlinear_benchmarks, dataset_name):
        # Get the dataset class and prepare data
        dataset_cls = getattr(nonlinear_benchmarks, dataset_name)
        prepare_nonlinear_benchmarks_data(dataset_dir, dataset_cls, 0.1, dt)
    elif dataset_name in AVAILABLE_DATASETS.keys():
        # Download the dataset from the URL in AVAILABLE_DATASETS
        download_and_extract_dataset(
            AVAILABLE_DATASETS[dataset_name], dataset_name, dataset_dir
        )
        prepare_downloaded_data(
            AVAILABLE_DATASETS[dataset_name], dataset_name, dataset_dir
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Ensure it is either in nonlinear_benchmarks or AVAILABLE_DATASETS."
        )

    return True  # Data was downloaded/prepared


def prepare_ship(dataset_dir: str, dataset_name: str) -> None:
    # nothing to prepare here, already done when downloading
    pass


def prepare_msd(dataset_dir: str, dataset_name: str) -> None:
    tracker = get_aggregated_tracker()
    iid_names = ["coupled-msd_u-15_K-200_T-1500"]
    ood_names = [
        "coupled_msd_u-35_f-80-120_K-50_T-1200",
        "coupled_msd_u-30_f-80-120_K-50_T-1200",
        "coupled_msd_u-10_f-80-120_K-50_T-1200",
        "coupled_msd_u-15_f-80-120_K-50_T-1200",
        "coupled_msd_u-50_f-80-120_K-50_T-1200",
        "coupled_msd_u-25_f-80-120_K-50_T-1200",
        "coupled_msd_u-20_f-80-120_K-50_T-1200",
        "coupled_msd_u-05_f-80-120_K-50_T-1200",
        "coupled_msd_u-40_f-80-120_K-50_T-1200",
        "coupled_msd_u-45_f-80-120_K-50_T-1200",
    ]
    msd_iid_dir = os.path.join("Coupled mass-spring-damper system", "in-distribution")
    msd_ood_dir = os.path.join(
        "Coupled mass-spring-damper system", "out-of-distribution"
    )

    train_processed_dir = os.path.join(
        dataset_dir, PROCESSED_FOLDER_NAME, TRAIN_FOLDER_NAME
    )
    id_test_processed_dir = os.path.join(
        dataset_dir, PROCESSED_FOLDER_NAME, IN_DISTRIBUTION_FOLDER_NAME
    )
    ood_test_processed_dir = os.path.join(
        dataset_dir, PROCESSED_FOLDER_NAME, OUT_OF_DISTRIBUTION_FOLDER_NAME
    )
    val_processed_dir = os.path.join(
        dataset_dir, PROCESSED_FOLDER_NAME, VALIDATION_FOLDER_NAME
    )

    # Check if the processed directories already exist
    if all(
        os.path.exists(dir)
        for dir in [
            train_processed_dir,
            id_test_processed_dir,
            ood_test_processed_dir,
            val_processed_dir,
        ]
    ):
        tracker.track(
            ev.Log("", "Processed directories already exist. Skipping preparation.")
        )
        return

    # handle id data
    for iid_name in iid_names:
        train_dir = os.path.join(
            dataset_dir,
            RAW_FOLDER_NAME,
            msd_iid_dir,
            iid_name,
            PROCESSED_FOLDER_NAME,
            TRAIN_FOLDER_NAME,
        )
        id_test_dir = os.path.join(
            dataset_dir,
            RAW_FOLDER_NAME,
            msd_iid_dir,
            iid_name,
            PROCESSED_FOLDER_NAME,
            "test",
        )
        val_dir = os.path.join(
            dataset_dir,
            RAW_FOLDER_NAME,
            msd_iid_dir,
            iid_name,
            PROCESSED_FOLDER_NAME,
            VALIDATION_FOLDER_NAME,
        )

        # Copy all CSV files from train_dir, id_test_dir, val_dir into their respective processed directories
        for src_dir, dest_dir in [
            (train_dir, train_processed_dir),
            (id_test_dir, id_test_processed_dir),
            (val_dir, val_processed_dir),
        ]:
            if os.path.exists(src_dir):
                os.makedirs(dest_dir, exist_ok=True)
            for file in os.listdir(src_dir):
                if file.endswith(".csv"):
                    src_file = os.path.join(src_dir, file)
                    dest_file = os.path.join(dest_dir, file)
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
    tracker.track(ev.Log("", "IID data preparation complete."))

    # handle ood data
    for ood_name in ood_names:
        ood_test_dir = os.path.join(
            dataset_dir,
            RAW_FOLDER_NAME,
            msd_ood_dir,
            ood_name,
            PROCESSED_FOLDER_NAME,
            "test",
        )

        # Copy all CSV files from ood_test_dir into the ood_test_processed_dir
        if os.path.exists(ood_test_dir):
            os.makedirs(ood_test_processed_dir, exist_ok=True)
        for file in os.listdir(ood_test_dir):
            if file.endswith(".csv"):
                src_file = os.path.join(ood_test_dir, file)
                dest_file = os.path.join(
                    ood_test_processed_dir,
                    f"{os.path.splitext(file)[0]}_{ood_name}.csv",
                )
                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)

    tracker.track(ev.Log("", "OOD data preparation complete."))


def prepare_hyst(dataset_dir: str, dataset_name: str) -> None:
    """
    Prepare HYST dataset into CSV files.

    Args:
        dataset_dir: Directory to save the prepared data
    """
    raw_dir = os.path.join(dataset_dir, RAW_FOLDER_NAME)
    if not os.path.exists(raw_dir):
        raise ValueError(
            f"Raw data directory '{raw_dir}' does not exist. Please download the dataset first."
        )

    # Create a directory called "processed" on the same level as "raw"
    processed_dir = os.path.join(dataset_dir, PROCESSED_FOLDER_NAME)
    os.makedirs(processed_dir, exist_ok=True)

    # Define file paths
    # training
    uy_train_filepaths = [os.path.join(raw_dir, "hyst_train.mat")]
    # test
    types = ["multisine", "sinesweep"]
    test_root_dir = os.path.join(
        raw_dir, "BoucWenFiles", "Test signals", "Validation signals"
    )
    u_test_filepaths = [
        os.path.join(test_root_dir, f"uval_{type}.mat") for type in types
    ]
    y_test_filepaths = [
        os.path.join(test_root_dir, f"yval_{type}.mat") for type in types
    ]

    # Load training data
    if not all(os.path.exists(path) for path in uy_train_filepaths):
        raise ValueError("Training data files are missing in the raw directory.")

    train_u = loadmat(uy_train_filepaths[0])["u"].flatten()
    train_y = loadmat(uy_train_filepaths[0])["y"].flatten()
    # Create DataFrame
    # Split 10% of the training data for validation
    split_idx = int(len(train_u) * 0.9)
    train_u_split, val_u_split = train_u[:split_idx], train_u[split_idx:]
    train_y_split, val_y_split = train_y[:split_idx], train_y[split_idx:]

    # Create DataFrames for training and validation
    train_df = pd.DataFrame({"u_1": train_u_split, "y_1": train_y_split})
    val_df = pd.DataFrame({"u_1": val_u_split, "y_1": val_y_split})

    # Load test data
    if not all(os.path.exists(path) for path in u_test_filepaths + y_test_filepaths):
        raise ValueError("Test data files are missing in the raw directory.")

    test_u = [
        loadmat(path)[f"uval_{type}"].flatten()
        for path, type in zip(u_test_filepaths, types)
    ]
    test_y = [
        loadmat(path)[f"yval_{type}"].flatten()
        for path, type in zip(y_test_filepaths, types)
    ]

    # Create DataFrames
    train_df = pd.DataFrame({"u_1": train_u, "y_1": train_y})

    # Save to CSV
    train_dir = os.path.join(processed_dir, TRAIN_FOLDER_NAME)
    val_dir = os.path.join(processed_dir, VALIDATION_FOLDER_NAME)
    test_dir = os.path.join(processed_dir, IN_DISTRIBUTION_FOLDER_NAME)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_df.to_csv(os.path.join(train_dir, f"{dataset_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(val_dir, f"{dataset_name}_validation.csv"), index=False)
    for i, (u, y, type) in enumerate(zip(test_u, test_y, types)):
        test_df = pd.DataFrame({"u_1": u, "y_1": y})
        test_df.to_csv(
            os.path.join(test_dir, f"{dataset_name}_test_{type}.csv"), index=False
        )

    print(f"✅ Prepared HYST dataset with 1 train and 1 test sequence")


def save_to_csv(data: InputOutputList, folder_name: str, file_path: str, input_names: List[str]=[], output_names: List[str]=[]) -> None:
    """
    Save InputOutputList data to a CSV file.

    Args:
        data: InputOutputList containing input and output data
        file_path: Path to the CSV file
    """
    for idx, (input, output) in enumerate(zip(data.d, data.e)):
        if input_names and output_names:
            input_columns = input_names
            output_columns = output_names

        else:
            input_columns = [f"u_{i+1}" for i in range(input.shape[1])]
            output_columns = [f"y_{i+1}" for i in range(output.shape[1])]
        columns = input_columns + output_columns

        flattened_data = np.hstack([input, output])
        df = pd.DataFrame(flattened_data, columns=columns)

        indexed_file_path = f"{file_path}_{idx}.csv"
        df.to_csv(os.path.join(folder_name, indexed_file_path), index=False)


def prepare_downloaded_data(
    url: Dict[str, Dict[str, str]], dataset_name: str, dataset_dir: str
) -> None:
    """
    Prepare downloaded dataset into CSV files.

    Args:
        url: Dictionary containing dataset information, including relative file paths if archived.
        dataset_name: Name of the dataset (used for filename).
        dataset_dir: Directory to save the prepared data.
    """
    # Dynamically call the appropriate preparation function based on the dataset name
    prepare_function_name = f"prepare_{dataset_name}"
    if prepare_function_name in globals():
        prepare_function = globals()[prepare_function_name]
        prepare_function(dataset_dir, dataset_name)
    else:
        raise ValueError(
            f"No preparation function found for dataset '{dataset_name}'. "
            f"Please implement a function named '{prepare_function_name}' to handle this dataset."
        )


def prepare_nonlinear_benchmarks_data(
    dataset_dir: str, dataset_cls: Callable, val_split: float, dt: float
) -> None:
    """
    Download and prepare data from nonlinear_benchmarks into CSV files.

    Args:
        dataset_dir: Directory to save the prepared data
        dataset_cls: The nonlinear_benchmarks dataset class
        val_split: Fraction of training data to use for validation
    """
    # Load the raw data
    try:
        train_val, test = dataset_cls(
            atleast_2d=True, always_return_tuples_of_datasets=True
        )
    except Exception as e:
        print(f"Error loading dataset from nonlinear benchmark: {e}")
        raise ValueError(
            f"Could not load dataset '{dataset_cls.__name__}' from nonlinear_benchmarks: {e}"
        )

    # Check sampling time
    sampling_time = train_val[0].sampling_time
    if not np.isclose(sampling_time, dt):
        raise ValueError(
            f"Sampling time mismatch: dataset sampling time is {sampling_time}, but dt is {dt}."
        )

    # Check if all sequences have the same data length
    data_lengths = [len(seq.u) for seq in train_val + test]
    if len(set(data_lengths)) > 1:
        print(
            "⚠️ Warning: Not all sequences have the same data length. Proceeding with caution."
        )
    datasetname = dataset_cls.__name__

    processed_directory = os.path.join(dataset_dir, PROCESSED_FOLDER_NAME)

    # Get dimensions
    N_train_val, ne = train_val[0].y.shape
    _, nd = train_val[0].u.shape
    M_train_val, M_test = len(train_val), len(test)

    # Generate column names
    input_names = [f"u_{idx + 1}" for idx in range(nd)]
    output_names = [f"y_{idx + 1}" for idx in range(ne)]

    # Process training/validation data
    for m in range(M_train_val):
        write_train_val_data_to_csv_file(
            train_val[m],
            val_split,
            m,
            datasetname,
            processed_directory,
            input_names,
            output_names,
            nd,
            ne,
        )

    # Process test data
    for m in range(M_test):
        write_test_data_to_csv_file(
            test[m],
            m,
            datasetname,
            processed_directory,
            input_names,
            output_names,
            nd,
            ne,
        )

    print(
        f"✅ Prepared dataset '{datasetname}' with {M_train_val} train/val and {M_test} test sequences"
    )


def load_data(
    input_names: List[str],
    output_names: List[str],
    type: str,
    dataset_dir: str,
) -> InputOutputList:
    """
    Load data from CSV files in the specified type directory.

    This function only loads data and assumes the data has already been prepared.
    Use download_and_prepare_data() first if the dataset needs to be downloaded.

    Args:
        input_names: List of input column names to load
        output_names: List of output column names to load
        type: The type of data to load ('train', 'test', 'validation')
        dataset_dir: Directory containing the prepared data

    Returns:
        InputOutputList: Loaded input and output data

    Raises:
        ValueError: If the data directory doesn't exist and needs to be prepared first
    """
    type_path = os.path.join(dataset_dir, type)

    if not os.path.exists(type_path):
        # Check if this is a nonlinear_benchmarks dataset that needs preparation
        dataset_name = utils.get_dataset_name(dataset_dir)

        if hasattr(nonlinear_benchmarks, dataset_name):
            raise ValueError(
                f"Dataset directory '{type_path}' does not exist. "
                f"Please run download_and_prepare_data('{dataset_dir}') first to prepare the '{dataset_name}' dataset."
            )
        else:
            raise ValueError(
                f"Dataset directory '{type_path}' does not exist and '{dataset_name}' is not available in nonlinear_benchmarks."
            )

    return load_data_from_folder(type_path, input_names, output_names)


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

    directory = os.path.join(dataset_dir, IN_DISTRIBUTION_FOLDER_NAME)
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

    for data, t in zip(
        [train_data, val_data], [TRAIN_FOLDER_NAME, VALIDATION_FOLDER_NAME]
    ):
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
    if isinstance(model, base.DynamicIdentificationModel):
        torch.save(model.state_dict(), f"{file_path}.pth")


def save_model_parameter(
    model: base.DynamicIdentificationModel, file_name: str
) -> None:
    if isinstance(model, base.DynamicIdentificationModel):
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
                    torch.tensor(initialization["ss"]["dt"]),
                ),
                "transient_time": initialization["transient_time"],
                "window": initialization["window"],
                "horizon": initialization["horizon"],
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
            segment = traj[i * segment_length : (i + 1) * segment_length]
            segments.append(segment)
    return segments


def compute_energy(u_traj: np.ndarray) -> float:
    """Compute energy of a signal as the squared L2 norm."""
    if len(u_traj) == 0:
        raise ValueError("Signal must have a length greater than 0.")
    return float(np.linalg.norm(u_traj, ord=2) ** 2)


def process_data(
    u: List[np.ndarray], y: List[np.ndarray], w: int, h: int, seed: int = 42
) -> Tuple[
    Tuple[List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray]],
]:
    """
    Processes trajectory data by splitting, computing energy, and separating high energy sequences.

    Returns:
        Tuple containing d_ood, training_set, validation_set
    """
    segment_length = w + h + 1

    u_segments = split_trajectories(u, segment_length)
    y_segments = split_trajectories(y, segment_length)

    # Compute energy for each segment pair
    input_energies = [compute_energy(u_seg) for u_seg in u_segments]

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
    remaining_output = [y_segments[i] for i in remaining_indices]

    # Shuffle and split remaining data
    d_train, d_val, e_train, e_val = train_test_split(
        remaining_input,
        remaining_output,
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
    output_dir: str,
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
        print("ood already exists.")
        return
    os.makedirs(output_dir, exist_ok=True)
    num_trajectories = len(ood_input)
    assert num_trajectories == len(
        ood_output
    ), "Input and output lists must be the same length."

    for i, (u_traj, y_traj) in enumerate(zip(ood_input, ood_output)):
        if u_traj.shape[0] != y_traj.shape[0]:
            raise ValueError(
                f"Trajectory {i} has mismatched time steps: input {u_traj.shape[0]} vs output {y_traj.shape[0]}"
            )
        if u_traj.shape[1] != len(input_columns):
            raise ValueError(
                f"Input trajectory {i} has shape {u_traj.shape}, expected {len(input_columns)} input columns."
            )
        if y_traj.shape[1] != len(output_columns):
            raise ValueError(
                f"Output trajectory {i} has shape {y_traj.shape}, expected {len(output_columns)} output columns."
            )

        # Concatenate input and output signals
        combined = np.concatenate([u_traj, y_traj], axis=1)
        combined_columns = input_columns + output_columns

        df = pd.DataFrame(combined, columns=combined_columns)

        filepath = os.path.join(output_dir, f"traj_{i:03d}.csv")
        df.to_csv(filepath, index=False)

    print(
        f"✅ Saved {num_trajectories} combined input/output trajectory files to '{output_dir}'"
    )

def load_preprocessing_results(
    directory: str
) -> Optional[PreprocessingResults]:
    """
    Load preprocessing results from saved files or return None if not available.

    Args:
        result_directory: Directory where preprocessing results are saved
        experiment_config: Experiment configuration
        dataset_dir: Dataset directory

    Returns:
        PreprocessingResults or None: Preprocessing results if available, None otherwise
    """
    init_data = load_initialization(directory)
    normalization: Optional[Normalization] = None

    try:
        normalization = load_normalization(directory)
    except (FileNotFoundError, OSError):
        # Normalization data not found
        pass

    if init_data.data and normalization:

        return PreprocessingResults(
            system_identification=SystemIdentificationResults(
                ss=init_data.data["ss"], transient_time=init_data.data["transient_time"]
            ),
            horizon=init_data.data["horizon"],
            window=init_data.data["window"],
            normalization=normalization,
        )

    return None
