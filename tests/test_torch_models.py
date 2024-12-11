import sys
import os
import torch

from utils import run_tests

torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    root_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
    result_directory = os.path.join(root_dir, "_tmp/results")
    configuration_file = os.path.join(root_dir, "config/config_torch.json")
    data_directory = os.path.join(root_dir, "data")
    run_tests(configuration_file, data_directory, result_directory)
