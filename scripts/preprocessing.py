#!/usr/bin/env python3
"""
Preprocessing script for CRNN training data.

This script handles all preprocessing steps that need to be done before training:
1. N4SID system identification for transient time calculation
2. Horizon and window length calculation
3. Data normalization parameter calculation
4. Data splitting (to be implemented)

Usage:
    python preprocessing.py <model_name> <experiment_name> [--gpu]
    
Environment variables:
    CONFIGURATION: Path to configuration file
    DATASET_DIRECTORY: Path to dataset directory
    RESULT_DIRECTORY: Path to result base directory
"""

import argparse
import os

import torch

import crnn.configuration.base as cfg
from crnn.preprocessing.base import preprocess

torch.set_default_dtype(torch.double)


def main() -> None:
    """Command line interface for preprocessing."""
    parser = argparse.ArgumentParser(description="Run CRNN preprocessing")
    model_name, experiment_name, gpu = cfg.parse_input(parser)

    result_base_directory = os.path.expanduser(os.getenv(cfg.RESULT_DIR_ENV_VAR))
    dataset_dir = os.path.expanduser(os.getenv(cfg.DATASET_DIR_ENV_VAR))
    config_file_name = os.path.expanduser(os.getenv(cfg.CONFIG_FILE_ENV_VAR))

    results = preprocess(
        config_file_name,
        dataset_dir,
        result_base_directory,
        model_name,
        experiment_name,
    )
    
    print("Preprocessing completed successfully!")
    print(f"Transient time: {results.system_identification.transient_time}")
    print(f"Horizon: {results.horizon}, Window: {results.window}")
    if results.system_identification.fit is not None:
        print(f"N4SID fit: {results.system_identification.fit:.4f}")


if __name__ == "__main__":
    main()
