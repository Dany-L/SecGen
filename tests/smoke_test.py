from crnn.train import train
from crnn.evaluate import evaluate
from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
import sys
import os
import torch
import json
import itertools

torch.set_default_dtype(torch.double)

root_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
result_directory = os.path.join(root_dir, "_tmp/results")
configuration_file = os.path.join(root_dir, "config.json")
data_directory = os.path.join(root_dir, "data")

with open(configuration_file, mode="r") as f:
    config_dict = json.load(f)

config = ExperimentConfig.from_template(ExperimentTemplate(**config_dict))
model_names = config.m_names
experiment_names = list(config.experiments.keys())

par = itertools.product(model_names,[True, False])

for model_name, gpu in par:
    for experiment_name in experiment_names:
        train(
            configuration_file,
            data_directory,
            result_directory,
            model_name,
            experiment_name,
            gpu
        )
        evaluate(
            configuration_file,
            data_directory,
            result_directory,
            model_name,
            experiment_name,
        )
