# utils.py
from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
from typing import Dict, Any
from crnn.train.base import train, continue_training
from crnn.evaluate import evaluate
from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
from crnn.utils.base import get_device
import itertools
import json


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
