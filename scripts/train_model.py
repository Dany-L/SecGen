import argparse

import numpy as np
import torch
import os
from torch.utils.data import DataLoader

import crnn.configuration as cfg
from crnn.train import train

torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CRNN model.")
    model_name, experiment_name = cfg.parse_input(parser)

    result_base_directory = os.path.expanduser(os.getenv(cfg.RESULT_DIR_ENV_VAR))
    dataset_dir = os.path.expanduser(os.getenv(cfg.DATASET_DIR_ENV_VAR))
    config_file_name = os.path.expanduser(os.getenv(cfg.CONFIG_FILE_ENV_VAR))
    
    train(config_file_name, dataset_dir, result_base_directory, model_name, experiment_name)
