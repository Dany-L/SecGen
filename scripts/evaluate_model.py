import argparse
import crnn.tracker as base_tracker
from crnn import configuration as cfg
import os
from crnn.datasets import RecurrentWindowHorizonDataset
from crnn import io as io_handling
from torch.utils.data import DataLoader
from crnn.models import sector_bounded as models
from crnn.models import base as base_models
from crnn.evaluate import evaluate_model
from crnn.metrics import Rmse
import torch
from crnn.models.io import get_model_from_config

torch.set_default_dtype(torch.double)

def main() -> None:
    parser = argparse.ArgumentParser(description='Train a CRNN model.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    args = parser.parse_args()
    model_name = args.model_name
    result_directory = io_handling.get_result_directory_name(model_name)

    tracker = base_tracker.BaseTracker(result_directory, model_name, 'validation')
    base_config = cfg.load_configuration()
    test_inputs, test_outputs = io_handling.load_data(base_config.input_names,base_config.output_names,'test')
    test_dataset = RecurrentWindowHorizonDataset(test_inputs,test_outputs,base_config.horizons.testing, base_config.window)

    model = get_model_from_config(model_name,base_config,tracker)

    model = base_models.load_model(model, os.path.join(result_directory,tracker.get_model_filename()))

    evaluate_model(model,test_dataset,Rmse,tracker)

if __name__ == '__main__':
    main()