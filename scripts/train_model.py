import argparse

import torch
from torch.utils.data import DataLoader

import crnn.configuration as cfg
import crnn.io as io_handling
import crnn.tracker as base_tracker
import crnn.utils.plot as plot
from crnn.datasets import RecurrentWindowHorizonDataset
from crnn.loss import get_loss_function
from crnn.models.io import get_model_from_config
from crnn.optimizer import get_optimizer
from crnn.train import train_model

torch.set_default_dtype(torch.double)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CRNN model.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to train"
    )
    args = parser.parse_args()
    model_name: str = args.model_name

    result_directory = io_handling.create_result_directory(model_name)
    tracker = base_tracker.BaseTracker(result_directory, model_name)
    base_config = cfg.load_configuration()
    train_inputs, train_outputs = io_handling.load_data(
        base_config.input_names, base_config.output_names, "train"
    )
    train_dataset = RecurrentWindowHorizonDataset(
        train_inputs, train_outputs, base_config.horizons.training, base_config.window
    )
    train_loader = DataLoader(train_dataset, base_config.batch_size, drop_last=True)

    model = get_model_from_config(model_name, base_config, tracker)
    model.add_semidefinite_constraints()

    opt = get_optimizer(base_config.optimizer, model.parameters())
    loss_fcn = get_loss_function(base_config.loss_function)

    model = train_model(model, train_loader, base_config.epochs, opt, loss_fcn, tracker)
    tracker.track(base_tracker.SaveModel("", model))
    tracker.track(base_tracker.SaveModelParameter("", model))
    tracker.track(base_tracker.SaveConfig("", base_config))


if __name__ == "__main__":
    main()
