from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from . import configuration as cfg
from . import tracker as base_tracker
from .data_io import get_result_directory_name, load_data
from .datasets import get_datasets, get_loaders
from .loss import get_loss_function
from .models import base
from .models.model_io import get_model_from_config
from .optimizer import get_optimizer
from .utils import base as utils
from .utils import plot


def train(
    config_file_name: str,
    dataset_dir: str,
    result_base_directory: str,
    model_name: str,
    experiment_name: str,
) -> None:
    result_directory = get_result_directory_name(
        result_base_directory, model_name, experiment_name
    )
    tracker = base_tracker.BaseTracker(result_directory, model_name)

    config = cfg.load_configuration(config_file_name)
    experiment_config = config.experiments[experiment_name]
    model_config = config.models[f"{experiment_name}-{model_name}"]

    train_inputs, train_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        "train",
        dataset_dir,
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

    loaders = get_loaders(
        get_datasets(
            n_train_inputs,
            n_train_outputs,
            experiment_config.horizons.training,
            experiment_config.window,
        ),
        experiment_config.batch_size,
    )

    initializer, predictor = get_model_from_config(model_config)

    optimizers = get_optimizer(
        experiment_config, (initializer.parameters(), predictor.parameters())
    )
    loss_fcn = get_loss_function(experiment_config.loss_function)

    if experiment_config.initial_hidden_state == "zero":
        (initializer, predictor) = train_zero(
            models=(initializer, predictor),
            loaders=loaders,
            epochs=experiment_config.epochs,
            optimizers=optimizers,
            loss_function=loss_fcn,
            tracker=tracker,
        )
    elif experiment_config.initial_hidden_state == "joint":
        (initializer, predictor) = train_joint(
            models=(initializer, predictor),
            loaders=loaders,
            epochs=experiment_config.epochs,
            optimizers=optimizers,
            loss_function=loss_fcn,
            tracker=tracker,
        )
    elif experiment_config.initial_hidden_state == "separate":
        (initializer, predictor) = train_separate(
            models=(initializer, predictor),
            loaders=loaders,
            epochs=experiment_config.epochs,
            optimizers=optimizers,
            loss_function=loss_fcn,
            tracker=tracker,
        )
    else:
        raise ValueError(
            f"Unknown initial_hidden_state: {experiment_config.initial_hidden_state}"
        )
    tracker.track(base_tracker.SaveModel("", predictor, "predictor"))
    if initializer is not None:
        tracker.track(base_tracker.SaveModel("", initializer, "initializer"))
    tracker.track(base_tracker.SaveModelParameter("", predictor))
    tracker.track(base_tracker.SaveConfig("", "experiment", experiment_config))
    tracker.track(base_tracker.SaveConfig("", "model", model_config.parameters))


def train_joint(
    models: Tuple[base.ConstrainedModule],
    loaders: Tuple[DataLoader],
    epochs: int,
    optimizers: Tuple[Optimizer],
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    _, train_loader = loaders
    initializer, predictor = models
    opt_init, opt_pred = optimizers
    initializer.initialize_parameters()
    initializer.set_lure_system()
    problem_status = predictor.initialize_parameters()
    tracker.track(base_tracker.Log("", f"Initialize parameters: {problem_status}"))
    predictor.set_lure_system()
    predictor.train()
    tracker.track(base_tracker.Start(""))

    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            initializer.zero_grad()
            e_hat_init, h0 = initializer.forward(batch["d_init"])
            e_hat, _ = predictor.forward(batch["d"], h0)
            batch_loss_predictor = loss_function(e_hat, batch["e"])
            batch_loss_initializer = loss_function(
                e_hat_init[:, -1, :], batch["e_init"][:, -1, :]
            )
            batch_loss = batch_loss_predictor + batch_loss_initializer
            batch_loss.backward()
            opt_pred.step()
            predictor.set_lure_system()
            initializer.set_lure_system()
            loss += batch_loss.item()
        if epoch % 50 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            problem_status = predictor.project_parameters()
            tracker.track(
                base_tracker.Log("", f"Projecting parameters: {problem_status}")
            )
            predictor.set_lure_system()
        tracker.track(base_tracker.Log("", f"{epoch}/{epochs}\t l= {loss:.2f}"))
    tracker.track(base_tracker.Stop(""))

    return (initializer, predictor)


def train_zero(
    models: Tuple[base.ConstrainedModule],
    loaders: Tuple[DataLoader],
    epochs: int,
    optimizers: Tuple[Optimizer],
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    _, train_loader = loaders
    _, predictor = models
    _, opt_pred = optimizers
    predictor.initialize_parameters()
    predictor.set_lure_system()
    predictor.train()
    tracker.track(base_tracker.Start(""))

    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            e_hat, _ = predictor.forward(batch["d"])
            batch_loss = loss_function(e_hat, batch["e"])
            batch_loss.backward()
            opt_pred.step()
            predictor.set_lure_system()
            loss += batch_loss.item()
        if epoch % 50 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            problem_status = predictor.project_parameters()
            tracker.track(
                base_tracker.Log("", f"Projecting parameters: {problem_status}")
            )
            predictor.set_lure_system()
        tracker.track(base_tracker.Log("", f"{epoch}/{epochs}\t l= {loss:.2f}"))
    tracker.track(base_tracker.Stop(""))

    return (None, predictor)


def train_separate(
    models: Tuple[base.ConstrainedModule],
    loaders: Tuple[DataLoader],
    epochs: int,
    optimizers: Tuple[Optimizer],
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    init_loader, pred_loader = loaders
    initializer, predictor = models
    opt_init, opt_pred = optimizers
    initializer.initialize_parameters()
    initializer.set_lure_system()
    initializer.train()
    problem_status = predictor.initialize_parameters()
    tracker.track(base_tracker.Log("", f"Initialize parameters: {problem_status}"))
    predictor.set_lure_system()
    predictor.train()
    tracker.track(base_tracker.Start(""))

    # initializer
    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(init_loader):
            initializer.zero_grad()
            e_hat_init, _ = initializer.forward(batch["d"])
            batch_loss = loss_function(e_hat_init[:, -1, :], batch["e"][:, -1, :])
            batch_loss.backward()
            opt_init.step()
            initializer.set_lure_system()
            loss += batch_loss.item()

        tracker.track(
            base_tracker.Log("", f"{epoch}/{epochs} (initializer)\t l= {loss:.2f}")
        )

    # predictor
    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(pred_loader):
            predictor.zero_grad()
            initializer.zero_grad()
            with torch.no_grad():
                _, h0 = initializer.forward(batch["d_init"])
            e_hat, _ = predictor.forward(batch["d"], h0)
            batch_loss = loss_function(e_hat, batch["e"])
            batch_loss.backward()
            opt_pred.step()
            predictor.set_lure_system()
            loss += batch_loss.item()

        if epoch % 50 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            problem_status = predictor.project_parameters()
            tracker.track(
                base_tracker.Log("", f"Projecting parameters: {problem_status}")
            )
            predictor.set_lure_system()

        tracker.track(
            base_tracker.Log("", f"{epoch}/{epochs} (predictor)\t l= {loss:.2f}")
        )
    tracker.track(base_tracker.Stop(""))

    return (initializer, predictor)
