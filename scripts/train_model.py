import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import crnn.configuration as cfg
import crnn.tracker as base_tracker
import crnn.utils.base as utils
import crnn.utils.plot as plot
from crnn.data_io import create_result_directory, load_data
from crnn.datasets import RecurrentWindowHorizonDataset
from crnn.loss import get_loss_function
from crnn.models.model_io import get_model_from_config
from crnn.optimizer import get_optimizer
from crnn.train import train_joint, train_separate, train_zero

torch.set_default_dtype(torch.double)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CRNN model.")
    parser.add_argument("model", type=str, help="Name of the model to train")
    args = parser.parse_args()
    model_name: str = args.model

    result_directory = create_result_directory(model_name)
    tracker = base_tracker.BaseTracker(result_directory, model_name)
    base_config = cfg.load_configuration()
    train_inputs, train_outputs = load_data(
        base_config.input_names, base_config.output_names, "train"
    )
    input_mean, input_std = utils.get_mean_std(train_inputs)
    output_mean, output_std = utils.get_mean_std(train_outputs)
    n_train_inputs = utils.normalize(train_inputs, input_mean, input_std)
    n_train_outputs = utils.normalize(train_outputs, output_mean, output_std)
    tracker.track(
        base_tracker.SaveNormalization(
            "",
            input=base_tracker.NormalizationParameters(input_mean, input_std),
            output=base_tracker.NormalizationParameters(output_mean, output_std),
        )
    )
    assert (
        np.mean(np.vstack(n_train_inputs) < 1e-5)
        and np.std(np.vstack(n_train_inputs)) - 1 < 1e-5
    )
    assert (
        np.mean(np.vstack(n_train_outputs) < 1e-5)
        and np.std(np.vstack(n_train_outputs)) - 1 < 1e-5
    )

    n_train_dataset = RecurrentWindowHorizonDataset(
        n_train_inputs,
        n_train_outputs,
        base_config.horizons.training,
        base_config.window,
    )
    n_train_loader = DataLoader(
        n_train_dataset, base_config.batch_size, drop_last=True, shuffle=True
    )

    initializer, predictor = get_model_from_config(model_name, base_config, tracker)
    predictor.add_semidefinite_constraints()

    optimizers = get_optimizer(
        base_config.optimizer, (initializer.parameters(), predictor.parameters())
    )
    loss_fcn = get_loss_function(base_config.loss_function)

    if base_config.optimizer.initial_hidden_state == "zero":
        (initializer, predictor) = train_zero(
            models=(initializer, predictor),
            train_loader=n_train_loader,
            epochs=base_config.epochs,
            optimizers=optimizers,
            loss_function=loss_fcn,
            tracker=tracker,
        )
    elif base_config.optimizer.initial_hidden_state == "joint":
        (initializer, predictor) = train_joint(
            models=(initializer, predictor),
            train_loader=n_train_loader,
            epochs=base_config.epochs,
            optimizers=optimizers,
            loss_function=loss_fcn,
            tracker=tracker,
        )
    elif base_config.optimizer.initial_hidden_state == "separate":
        (initializer, predictor) = train_separate(
            models=(initializer, predictor),
            train_loader=n_train_loader,
            epochs=base_config.epochs,
            optimizers=optimizers,
            loss_function=loss_fcn,
            tracker=tracker,
        )
    else:
        raise ValueError(
            f"Unknown initial_hidden_state: {base_config.optimizer.initial_hidden_state}"
        )
    tracker.track(base_tracker.SaveModel("", predictor, "predictor"))
    tracker.track(base_tracker.SaveModel("", initializer, "initializer"))
    tracker.track(base_tracker.SaveModelParameter("", predictor))
    tracker.track(base_tracker.SaveConfig("", base_config))


if __name__ == "__main__":
    main()
