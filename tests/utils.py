# utils.py
from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
from typing import Dict, Any
from crnn.train.base import train, continue_training
from crnn.evaluate import evaluate
from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
from crnn.utils.base import get_device
import itertools
import json
import os
import pandas as pd
import numpy as np


def run_tests(
    configuration_file: str, data_directory: str, result_directory: str
) -> None:

    with open(configuration_file, mode="r") as f:
        config_dict = json.load(f)

    config = ExperimentConfig.from_template(ExperimentTemplate(**config_dict))
    model_names = config.m_names
    experiment_names = list(config.experiments.keys())
    
    if get_device(gpu=True) == "cuda":
        devices = [True,False]
    else:
        devices = [False]

    par = itertools.product(model_names, devices)

    for model_name, gpu in par:
        for experiment_name in experiment_names:
            train(
                configuration_file,
                data_directory,
                result_directory,
                model_name,
                experiment_name,
                gpu,
            )
            continue_training(
                configuration_file,
                data_directory,
                result_directory,
                model_name,
                experiment_name,
                gpu,
            )
            evaluate(
                configuration_file,
                data_directory,
                result_directory,
                model_name,
                experiment_name,
            )


def load_filtered_csvs(
    data_directory: str,
    positive_filter: list[str],
    exclude_filter: list[str],
    input_names: list[str],
    output_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate CSV files from data_directory that contain ALL positive_filter
    and do NOT contain ANY exclude_filter strings in their filenames.
    Returns concatenated numpy arrays for ds and es.
    """
    def file_matches(filename: str, pos: list[str], exc: list[str]) -> bool:
        return all(s in filename for s in pos) and not any(s in filename for s in exc)

    files = [
        f for f in os.listdir(data_directory)
        if file_matches(f, positive_filter, exclude_filter) and f.endswith(".csv")
    ]
    ds_list = [
        pd.read_csv(os.path.join(data_directory, f), usecols=input_names, header=0).values.T
        for f in files
    ]
    es_list = [
        pd.read_csv(os.path.join(data_directory, f), usecols=output_names, header=0).values.T
        for f in files
    ]
    ds = np.concatenate(ds_list, axis=1) if ds_list else np.empty((0, 0))
    es = np.concatenate(es_list, axis=1) if es_list else np.empty((0, 0))
    return ds, es