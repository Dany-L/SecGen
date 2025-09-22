"""
Preprocessing module for CRNN training data.

This module handles all preprocessing steps that need to be done before training:
1. N4SID system identification for transient time calculation
2. Horizon and window length calculation
3. Data normalization parameter calculation
4. Data splitting (to be implemented)
"""

import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from numpy.typing import NDArray


from ..train.base import load_preprocessing_results
from ..configuration.experiment import load_configuration
from ..configuration.base import (
    SystemIdentificationResults, 
    PreprocessingResults, 
    Normalization, 
    NormalizationParameters,
    IN_DISTRIBUTION_FOLDER_NAME,
    PROCESSED_FOLDER_NAME,
    OUT_OF_DISTRIBUTION_FOLDER_NAME,
    PREPARED_FOLDER_NAME,
    TRAIN_FOLDER_NAME,
    TEST_FOLDER_NAME,
    VALIDATION_FOLDER_NAME,
    InputOutputList
)
from ..io.data import (
    get_result_directory_name,
    load_data,
    load_initialization,
    download_and_prepare_data,
    save_to_csv
)
from ..models import base
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker, set_aggregated_tracker
from ..io.tracker import get_trackers_from_config
from ..utils import base as utils
from ..systemtheory.analysis import get_transient_time


def check_preprocessing_completed(result_directory: str) -> bool:
    """
    Check if preprocessing has been completed for a given result directory.
    
    Args:
        result_directory: Directory to check for preprocessing results
        
    Returns:
        bool: True if preprocessing is complete, False otherwise
    """
    try:
        # Check for initialization data
        init_file = os.path.join(result_directory, "initialization.pkl")
        normalization_file = os.path.join(result_directory, "normalization.pkl")
        
        return os.path.exists(init_file) and os.path.exists(normalization_file)
    except Exception:
        return False


def calculate_system_identification(
    train_inputs: List[NDArray[np.float64]], 
    train_outputs: List[NDArray[np.float64]], 
    dt: float, 
    nz: int, 
    tracker: AggregatedTracker
) -> SystemIdentificationResults:
    """
    Perform N4SID system identification and calculate transient time.
    
    Args:
        train_inputs: List of training input arrays
        train_outputs: List of training output arrays
        dt: Sampling time
        nz: Number of states for N4SID
        tracker: Tracker for logging
        
    Returns:
        SystemIdentificationResults: Contains system identification data
    """
    ss, fit, rmse = base.run_n4sid(
        train_inputs,
        train_outputs,
        dt=dt,
        nx=10, # fixed for now, should be cleverly chosen at some point
    )
    
    tracker.track(
        ev.Log(
            "",
            f"Evaluation of linear approximation on training data: Fit= {fit}, MSE= {rmse}",
        )
    )
    
    transient_time = int(np.max(get_transient_time(ss) / dt))
    tracker.track(ev.TrackParameter("", "transient_time", transient_time))
    tracker.track(ev.SaveInitialization("", ss, {"transient_time": transient_time}))
    
    return SystemIdentificationResults(
        ss=ss,
        transient_time=transient_time,
        fit=fit,
        rmse=rmse,
        N=train_inputs[0].shape[1]  # sequence length
    )


def calculate_horizon_and_window(transient_time: int, N: int, dt: float, window_ratio: float = 0.1, min_time:float=0.1) -> Tuple[int, int]:
    """
    Calculate prediction horizon and window length.
    
    Args:
        transient_time: Transient time from system identification
        N: Sequence length
        window_ratio: Ratio of window to horizon (default 0.1)
        
    Returns:
        Tuple of (horizon, window)
    """
    max_length = N // 2 - 1  # Ensure window + horizon + 1 is at most half of N
    # horizon = min(max(transient_time, int(min_time / dt)), max_length // int(1 / window_ratio))
    horizon = min(max(transient_time, int(min_time / dt)), 100)
    window = int(window_ratio * horizon)
    return horizon, window


def calculate_normalization_parameters(
    train_inputs: List[NDArray[np.float64]], 
    train_outputs: List[NDArray[np.float64]], 
    tracker: AggregatedTracker
) -> Normalization:
    """
    Calculate normalization parameters for inputs and outputs.
    
    Args:
        train_inputs: List of training input arrays
        train_outputs: List of training output arrays
        tracker: Tracker for logging
        
    Returns:
        Normalization: Contains normalization parameters
    """
    input_mean, input_std = utils.get_mean_std(train_inputs)
    output_mean, output_std = utils.get_mean_std(train_outputs)
    
    # Normalize training data for validation
    n_train_inputs = utils.normalize(train_inputs, input_mean, input_std)
    n_train_outputs = utils.normalize(train_outputs, output_mean, output_std)
    
    # Create normalization parameters
    normalization = Normalization(
        input=NormalizationParameters(mean=input_mean, std=input_std),
        output=NormalizationParameters(mean=output_mean, std=output_std)
    )
    
    # Save normalization parameters
    tracker.track(ev.SaveNormalization("", input=normalization.input, output=normalization.output))
    
    # Validation checks
    assert (
        np.mean(np.vstack(n_train_inputs)) < 1e-5
        and abs(np.std(np.vstack(n_train_inputs)) - 1) < 1e-5
    ), "Input normalization failed"
    
    assert (
        np.mean(np.vstack(n_train_outputs)) < 1e-5
        and abs(np.std(np.vstack(n_train_outputs)) - 1) < 1e-5
    ), "Output normalization failed"
    
    return normalization


def perform_data_splitting(
    train_data: InputOutputList,
    val_data: InputOutputList,
    test_data: InputOutputList,
    ood_test_data: InputOutputList,
    window: int,
    horizon: int,
    dataset_dir: str,
    split_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Perform data splitting (placeholder for future implementation).
    
    Args:
        train_inputs: Training input data
        train_outputs: Training output data
        val_inputs: Validation input data
        val_outputs: Validation output data
        window: Window length
        horizon: Prediction horizon
        split_config: Configuration for data splitting (future use)
        
    Returns:
        Dict containing split data (currently returns original data)
    """
    # TODO: Implement data splitting logic here
    # This could include:
    # - Energy-based splitting
    # - OOD detection and separation
    # - Custom splitting strategies

    # Ensure OOD folder exists
    ood_folder = os.path.join(dataset_dir, PREPARED_FOLDER_NAME, OUT_OF_DISTRIBUTION_FOLDER_NAME)
    id_folder = os.path.join(dataset_dir, PREPARED_FOLDER_NAME, IN_DISTRIBUTION_FOLDER_NAME)
    train_folder = os.path.join(dataset_dir, PREPARED_FOLDER_NAME, TRAIN_FOLDER_NAME)
    val_folder = os.path.join(dataset_dir, PREPARED_FOLDER_NAME, VALIDATION_FOLDER_NAME)
    os.makedirs(ood_folder, exist_ok=True)
    os.makedirs(id_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Split sequences into subsequences of length window + horizon + 1
    def split_into_subsequences(data: InputOutputList, seq_length: int) -> InputOutputList:
        inputs, outputs = [], []
        for d, e in zip(data.d, data.e):
            num_subsequences = d.shape[0] // seq_length
            for i in range(num_subsequences):
                start_idx = i * seq_length
                inputs.append(d[start_idx:start_idx + seq_length])
                outputs.append(e[start_idx:start_idx + seq_length])
        return InputOutputList(inputs, outputs)

    train_data = split_into_subsequences(train_data, window + horizon + 1)
    val_data = split_into_subsequences(val_data, window + horizon + 1)
    test_data = split_into_subsequences(test_data, window + horizon + 1)
    # Check if OOD test data is empty
    if not ood_test_data.d:
        

    # Calculate signal energy for inputs
        def compute_energy(data: InputOutputList) -> List[float]:
            return [np.sum(np.square(d)) for d in data.d]

        train_energies = compute_energy(train_data)
        test_energies = compute_energy(test_data)
        val_energies = compute_energy(val_data)

        # Separate the top 10% highest energy inputs and corresponding outputs
        def separate_high_energy(data: InputOutputList, energies: List[float], top_percentage: float) -> Tuple[InputOutputList, InputOutputList]:
            threshold = np.percentile(energies, 100 * (1 - top_percentage))
            high_energy_inputs, high_energy_outputs = [], []
            remaining_inputs, remaining_outputs = [], []

            for d, e, energy in zip(data.d, data.e, energies):
                if energy >= threshold:
                    high_energy_inputs.append(d)
                    high_energy_outputs.append(e)
                else:
                    remaining_inputs.append(d)
                    remaining_outputs.append(e)

            high_energy_data = InputOutputList(high_energy_inputs, high_energy_outputs)
            remaining_data = InputOutputList(remaining_inputs, remaining_outputs)
            return high_energy_data, remaining_data

        high_energy_train, train_data = separate_high_energy(train_data, train_energies, 0.1)
        high_energy_val, val_data = separate_high_energy(val_data, val_energies, 0.1)
        high_energy_test, test_data = separate_high_energy(test_data, test_energies, 0.1)

        for high_energy, split in zip(
            [high_energy_train, high_energy_val, high_energy_test],
            ["train", "validation", "test"]
        ):
            save_to_csv(high_energy, ood_folder, f"high_energy_{split}")
    else:
        ood_test_data = split_into_subsequences(ood_test_data, window + horizon + 1)
        save_to_csv(ood_test_data, ood_folder, 'ood_test')

    save_to_csv(test_data, id_folder, 'test')
    save_to_csv(val_data, val_folder, 'validation')
    save_to_csv(train_data, train_folder, 'train')

def preprocess(
    config_file_name: str, 
    dataset_dir: str, 
    result_base_directory: str, 
    model_name: str, 
    experiment_name: str, 
) -> PreprocessingResults:
    """
    Main preprocessing function.
    
    Args:
        config_file_name: Path to configuration file
        dataset_dir: Dataset directory
        result_base_directory: Base directory for results
        model_name: Model name
        experiment_name: Experiment name
        gpu: Whether to use GPU
        
    Returns:
        PreprocessingResults: Complete preprocessing results
    """
    # Setup directories and configuration
    result_directory = get_result_directory_name(
        result_base_directory, model_name, experiment_name
    )
    
    # Check if preprocessing already completed
    if check_preprocessing_completed(result_directory):
        print(f"Preprocessing already completed for {result_directory}")
        print("Loading existing preprocessing results...")
        
        # Load existing results
        config = load_configuration(config_file_name)
        experiment_config = config.experiments[experiment_name]
        
        existing_results = load_preprocessing_results(result_directory, experiment_config, dataset_dir)
        if existing_results is not None:
            return existing_results
        else:
            print("Warning: Preprocessing files exist but couldn't be loaded. Rerunning preprocessing...")
    
    config = load_configuration(config_file_name)
    experiment_config = config.experiments[experiment_name]
    full_model_name = f"{experiment_name}-{model_name}"
    trackers_config = config.trackers
    dataset_name = os.path.dirname(os.path.dirname(dataset_dir))
    
    if experiment_config.debug:
        import torch
        import pandas as pd
        torch.manual_seed(42)
    
    # Setup tracking
    trackers = get_trackers_from_config(
        trackers_config, result_directory, full_model_name, "preprocessing"
    )
    tracker = AggregatedTracker(trackers)
    tracker.track(ev.Start("", dataset_name))
    set_aggregated_tracker(tracker)

    # Log configuration parameters
    tracker.track(ev.TrackParameter("", "model_name", model_name))
    tracker.track(ev.TrackParameter("", "experiment_name", experiment_name))
    tracker.track(ev.TrackParameter("", "dataset_name", dataset_name))
    

    
    # Step 0: Ensure data is downloaded and prepared
    tracker.track(ev.Log("", "Checking if dataset needs to be downloaded/prepared..."))
    data_was_prepared = download_and_prepare_data(dataset_dir, experiment_config.dt)
    if data_was_prepared:
        tracker.track(ev.Log("", f"Dataset '{dataset_name}' was downloaded and prepared"))
    else:
        tracker.track(ev.Log("", f"Dataset '{dataset_name}' already exists"))

    # after this step we should have train, validation and test in the processed folder.
    processed_dataset_dir = os.path.join(dataset_dir, PROCESSED_FOLDER_NAME)

    # Load training data
    train_inputs, train_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        TRAIN_FOLDER_NAME,
        processed_dataset_dir,
    )
    N = train_inputs[0].shape[0]  # sequence length
    tracker.track(
        ev.Log(
            "",
            f"Training samples: {np.sum([train_input.shape[0] for train_input in train_inputs])}",
        )
    )

    # Step 1: System identification and transient time calculation
    init_data = load_initialization(result_directory)
    
    if not init_data.data:
        tracker.track(ev.Log("", "Performing N4SID system identification..."))
        system_id_results = calculate_system_identification(
            train_inputs, train_outputs, 
            experiment_config.dt, experiment_config.nz, tracker
        )
        init_data.data = {
            "ss": system_id_results.ss, 
            "transient_time": system_id_results.transient_time
        }
    else:
        tracker.track(ev.Log("", "Using existing system identification results"))
        system_id_results = SystemIdentificationResults(
            ss=init_data.data["ss"],
            transient_time=init_data.data["transient_time"]
        )
    
    # Step 2: Calculate horizon and window
    horizon, window = calculate_horizon_and_window(system_id_results.transient_time, N, init_data.data['ss'].dt)
    tracker.track(ev.Log("", f"Calculated window: {window}, horizon: {horizon}"))
    
    # Step 3: Calculate normalization parameters
    tracker.track(ev.Log("", "Calculating normalization parameters..."))
    normalization = calculate_normalization_parameters(
        train_inputs, train_outputs, tracker
    )

    # Load validation and test data
    val_inputs, val_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        VALIDATION_FOLDER_NAME,
        processed_dataset_dir,
    )
    test_inputs, test_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        IN_DISTRIBUTION_FOLDER_NAME,
        processed_dataset_dir,
    )
    try:
        ood_inputs, ood_outputs = load_data(
            experiment_config.input_names,
            experiment_config.output_names,
            OUT_OF_DISTRIBUTION_FOLDER_NAME,
            processed_dataset_dir,
        )
        ood_test_data = InputOutputList(ood_inputs, ood_outputs)
    except FileNotFoundError:
        tracker.track(ev.Log("", "No OOD data found, proceeding without it"))
        ood_test_data = InputOutputList([], [])
    train_data = InputOutputList(train_inputs, train_outputs)
    val_data = InputOutputList(val_inputs, val_outputs)
    test_data = InputOutputList(test_inputs, test_outputs)  

    # Step 4: Data splitting
    tracker.track(ev.Log("", "Performing data splitting..."))
    prepare_folder_name = os.path.join(dataset_dir, PREPARED_FOLDER_NAME)
    if not os.path.exists(prepare_folder_name):
        tracker.track(ev.Log("", f"Prepared data folder does not exist at {prepare_folder_name}, performing data splitting."))
        perform_data_splitting(
            train_data, val_data, test_data, ood_test_data,
            window, horizon, dataset_dir
        )
    else:
        tracker.track(ev.Log("", f"Prepared data folder already exists at {prepare_folder_name}, skipping data splitting."))
    
    # Prepare results using dataclass
    results = PreprocessingResults(
        system_identification=system_id_results,
        horizon=horizon,
        window=window,
        normalization=normalization,
        dataset_info={
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir,
            "input_names": experiment_config.input_names,
            "output_names": experiment_config.output_names,
        }
    )
    
    tracker.track(ev.Log("", "Preprocessing completed successfully"))
    tracker.track(ev.TrackParameter("", "preprocessing_completed", True))
    tracker.track(ev.Stop(""))
    
    return results
