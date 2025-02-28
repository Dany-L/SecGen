import sys
import os
import torch
from crnn.configuration.base import IN_DISTRIBUTION_FOLDER_NAME, PROCESSED_FOLDER_NAME

from utils import run_tests

torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    data_directory_names = ['CED', 'Cascaded_Tanks', 'EMPS', 'Silverbox', 'WienerHammerBenchMark', 'ParWH', 'F16']

    for dataset_name in data_directory_names:
        print(f'--- Dataset {dataset_name} ---')
        root_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
        result_directory = os.path.join(root_dir, "_tmp/results")
        configuration_file = os.path.join(root_dir, f"config/nonlinear_benchmarks/{dataset_name}.json")
        data_directory = os.path.join(root_dir, "_tmp", "data", dataset_name, IN_DISTRIBUTION_FOLDER_NAME, PROCESSED_FOLDER_NAME)
        os.makedirs(data_directory, exist_ok=True)
        run_tests(configuration_file, data_directory, result_directory)