import argparse
import os

import numpy as np
import torch

import crnn.tracker as base_tracker
from crnn import configuration as cfg
from crnn.data_io import (get_result_directory_name, load_data,
                          load_normalization)
from crnn.datasets import RecurrentWindowHorizonDataset
from crnn.evaluate import evaluate_model
from crnn.metrics import Rmse
from crnn.models import base as base_models
from crnn.models.model_io import get_model_from_config
from crnn.utils import base as utils

torch.set_default_dtype(torch.double)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CRNN model.")
    parser.add_argument("model", type=str, help="Name of the model to train")
    args = parser.parse_args()
    model_name: str = args.model
    result_directory = get_result_directory_name(model_name)

    tracker = base_tracker.BaseTracker(result_directory, model_name, "validation")
    base_config = cfg.load_configuration()
    test_inputs, test_outputs = load_data(
        base_config.input_names, base_config.output_names, "test"
    )
    normalization = load_normalization(result_directory)
    n_test_inputs = utils.normalize(
        test_inputs, normalization.input.mean, normalization.input.std
    )

    test_dataset = RecurrentWindowHorizonDataset(
        n_test_inputs, test_outputs, base_config.horizons.testing, base_config.window
    )

    initializer, predictor = get_model_from_config(model_name, base_config, tracker)

    predictor = base_models.load_model(
        predictor,
        os.path.join(result_directory, tracker.get_model_file_name("predictor")),
    )
    initializer = base_models.load_model(
        initializer,
        os.path.join(result_directory, tracker.get_model_file_name("initializer")),
    )

    evaluate_model((initializer, predictor), test_dataset, Rmse, normalization, tracker)


if __name__ == "__main__":
    main()
