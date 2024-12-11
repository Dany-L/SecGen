import sys
import os
import torch
import time

from utils import run_tests

torch.set_default_dtype(torch.double)


def test() -> None:
    root_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
    result_directory = os.path.join(root_dir, "_tmp/results")
    configuration_file = os.path.join(root_dir, "config/config_jax.json")
    data_directory = os.path.join(root_dir, "data")
    run_tests(configuration_file, data_directory, result_directory)


if __name__ == "__main__":
    start_test = time.time()
    test()
    duration = time.time() - start_test
    print(f"--- Test duration: {time.strftime('%H:%M:%S', duration)} ---")
