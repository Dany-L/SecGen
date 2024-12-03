import os
import time
from typing import List, Optional, Tuple, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from .configuration.experiment import load_configuration, BaseExperimentConfig
from .data_io import get_result_directory_name, load_data
from .datasets import get_datasets, get_loaders
from .loss import get_loss_function
from .models import base
from .models.model_io import get_model_from_config
from .optimizer import get_optimizer
from .tracker import events as ev
from .tracker.base import AggregatedTracker, get_trackers_from_config
from .utils import base as utils
from .utils import plot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

PLOT_AFTER_EPOCHS: int = 100


def train(
    config_file_name: str,
    dataset_dir: str,
    result_base_directory: str,
    model_name: str,
    experiment_name: str,
    gpu: bool,
) -> None:
    result_directory = get_result_directory_name(
        result_base_directory, model_name, experiment_name
    )

    config = load_configuration(config_file_name)
    experiment_config = config.experiments[experiment_name]
    model_name = f"{experiment_name}-{model_name}"
    model_config = config.models[model_name]
    trackers_config = config.trackers
    dataset_name = os.path.basename(os.path.dirname(dataset_dir))
    if experiment_config.debug:
        torch.manual_seed(42)

    trackers = get_trackers_from_config(
        trackers_config, result_directory, model_name, "training"
    )
    tracker = AggregatedTracker(trackers)
    tracker.track(ev.Start("", dataset_name))

    device = utils.get_device(gpu)
    tracker.track(ev.TrackParameter("", "device", device))
    tracker.track(ev.TrackParameter("", "model_class", model_config.m_class))
    tracker.track(ev.Log("", f"Train model {model_name} on {device}."))
    tracker.track(
        ev.TrackParameters(
            "",
            utils.get_config_file_name("experiment", model_name),
            experiment_config.model_dump(),
        )
    )
    tracker.track(
        ev.TrackParameters(
            "",
            utils.get_config_file_name("model", model_name),
            model_config.parameters.model_dump(),
        )
    )

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
        ev.SaveNormalization(
            "",
            input=ev.NormalizationParameters(input_mean, input_std),
            output=ev.NormalizationParameters(output_mean, output_std),
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
    start_time = time.time()
    with torch.device(device):
        loaders = get_loaders(
            get_datasets(
                n_train_inputs,
                n_train_outputs,
                experiment_config.horizons.training,
                experiment_config.window,
            ),
            experiment_config.batch_size,
            device,
        )

        initializer, predictor = get_model_from_config(model_config)

        optimizers = get_optimizer(
            experiment_config, (initializer.parameters(), predictor.parameters())
        )
        schedulers = [
            lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)
            for optimizer in optimizers
        ]
        loss_fcn = get_loss_function(experiment_config.loss_function)

        if experiment_config.initial_hidden_state == "zero":
            (initializer, predictor) = train_zero(
                models=(initializer, predictor),
                loaders=loaders,
                exp_config=experiment_config,
                optimizers=optimizers,
                loss_function=loss_fcn,
                schedulers=schedulers,
                tracker=tracker,
            )
        elif experiment_config.initial_hidden_state == "joint":
            (initializer, predictor) = train_joint(
                models=(initializer, predictor),
                loaders=loaders,
                exp_config=experiment_config,
                optimizers=optimizers,
                loss_function=loss_fcn,
                schedulers=schedulers,
                tracker=tracker,
            )
        elif experiment_config.initial_hidden_state == "separate":
            (initializer, predictor) = train_separate(
                models=(initializer, predictor),
                loaders=loaders,
                exp_config=experiment_config,
                optimizers=optimizers,
                loss_function=loss_fcn,
                schedulers=schedulers,
                tracker=tracker,
            )
        else:
            raise ValueError(
                f"Unknown initial_hidden_state: {experiment_config.initial_hidden_state}"
            )
    stop_time = time.time()
    training_duration = utils.get_duration_str(start_time, stop_time)
    tracker.track(ev.Log("", f"Training duration: {training_duration}"))
    tracker.track(ev.TrackParameter("", "training_duration", training_duration))
    tracker.track(ev.SaveModel("", predictor, "predictor"))
    if initializer is not None:
        tracker.track(ev.SaveModel("", initializer, "initializer"))
    tracker.track(ev.SaveModelParameter("", predictor))
    tracker.track(
        ev.SaveTrackingConfiguration("", trackers_config, model_name, result_directory)
    )
    tracker.track(ev.Stop(""))


def train_joint(
    models: Tuple[base.ConstrainedModule],
    loaders: Tuple[DataLoader],
    exp_config: BaseExperimentConfig,
    optimizers: List[Optimizer],
    loss_function: nn.Module,
    schedulers: List[lr_scheduler.ReduceLROnPlateau],
    tracker: AggregatedTracker = AggregatedTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    _, train_loader = loaders
    initializer, predictor = models
    (opt_pred,) = optimizers
    (sch_pred,) = schedulers
    initializer.initialize_parameters()
    initializer.set_lure_system()
    problem_status = predictor.initialize_parameters()
    tracker.track(ev.Log("", f"Initialize parameters: {problem_status}"))
    predictor.set_lure_system()
    predictor.train()
    t, increase_rate, increase_after_epochs = (
        exp_config.t,
        exp_config.increase_rate,
        exp_config.increase_after_epochs,
    )

    for epoch in range(exp_config.epochs):
        loss, phi = 0.0, 0.0  # phi is barrier
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            initializer.zero_grad()
            e_hat_init, h0 = initializer.forward(batch["d_init"])
            e_hat, _ = predictor.forward(batch["d"], h0)
            batch_loss_predictor = loss_function(e_hat, batch["e"])
            batch_phi = predictor.get_phi(t)
            batch_loss_initializer = loss_function(
                e_hat_init[:, -1, :], batch["e_init"][:, -1, :]
            )
            batch_loss = batch_loss_predictor + batch_loss_initializer
            (batch_loss + batch_phi).backward()
            opt_pred.step()
            predictor.set_lure_system()
            initializer.set_lure_system()
            loss += batch_loss.item()
            phi += batch_phi.item()
        if epoch % PLOT_AFTER_EPOCHS == 0:
            fig = plot.plot_sequence(
                [
                    e_hat[0, :].cpu().detach().numpy(),
                    batch["e"][0, :].cpu().detach().numpy(),
                ],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(ev.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            problem_status = predictor.project_parameters()
            tracker.track(ev.Log("", f"Projecting parameters: {problem_status}"))
            predictor.set_lure_system()

        tracker.track(
            ev.Log("", f"{epoch}/{exp_config.epochs}\t l= {loss:.2f} \t phi= {phi:.2f}")
        )
        tracker.track(
            ev.TrackMetrics(
                "",
                {
                    "epoch.loss.predictor": float(loss),
                    "epoch.phi.predictor": float(phi),
                },
                epoch,
            )
        )

        sch_pred.step(loss)

        if (epoch + 1) % increase_after_epochs == 0:
            t = t * increase_rate
            tracker.track(
                ev.Log(
                    "",
                    f"Increase t by {increase_rate} to {t}, lr: {sch_pred.get_last_lr()}",
                )
            )

    return (initializer, predictor)


def train_zero(
    models: Tuple[base.ConstrainedModule],
    loaders: Tuple[DataLoader],
    exp_config: BaseExperimentConfig,
    optimizers: List[Optimizer],
    schedulers: List[lr_scheduler.ReduceLROnPlateau],
    loss_function: nn.Module,
    tracker: AggregatedTracker = AggregatedTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    _, train_loader = loaders
    _, predictor = models
    _, opt_pred = optimizers
    _, sch_pred = schedulers
    predictor.initialize_parameters()
    predictor.set_lure_system()
    predictor.train()
    t, increase_rate, increase_after_epochs = (
        exp_config.t,
        exp_config.increase_rate,
        exp_config.increase_after_epochs,
    )

    for epoch in range(exp_config.epochs):
        loss, phi = 0.0, 0.0  # phi is barrier
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            if exp_config.ensure_constrained_method == "armijo":
                opt_fcn = base.OptFcn(
                    batch["d"], batch["e"], predictor, t, l=loss_function
                )
                f = lambda theta: opt_fcn.f(theta)
                dF = lambda theta: opt_fcn.dF(theta).reshape((-1, 1))

                theta = torch.hstack(
                    [
                        p.flatten().clone()
                        for p in predictor.parameters()
                        if p is not None
                    ]
                ).reshape(-1, 1)

                old_theta = theta.clone()
                batch_loss, batch_phi = f(theta), torch.tensor(0.0)

                arm = Armijo(f, dF, 1)
                # if step == 0 and epoch % PLOT_AFTER_EPOCHS == 0:
                #     fig = arm.plot_step_size_function(old_theta)
                #     tracker.track(ev.SaveFig('', fig, f'step_size_plot-{epoch}'))
                dir = -dF(old_theta)
                s = arm.linesearch(old_theta, dir)
                new_theta = old_theta + s * dir
                opt_fcn.set_vec_pars_to_model(new_theta)

                tracker.track(ev.TrackMetrics("", {
                        "loss.step": float(batch_loss),
                        "stepsize.step": float(s),
                    },
                    epoch * len(train_loader) + step
                ))

            else:
                e_hat, _ = predictor.forward(batch["d"])
                batch_loss = loss_function(e_hat, batch["e"])
                batch_phi = predictor.get_phi(t)
                (batch_loss + batch_phi).backward()
                opt_pred.step()

            predictor.set_lure_system()
            loss += batch_loss.item()
            phi += batch_phi.item()

        if not predictor.check_constraints():
            problem_status = predictor.project_parameters()
            tracker.track(ev.Log("", f"Projecting parameters: {problem_status}"))
            predictor.set_lure_system()

        tracker.track(
            ev.Log("", f"{epoch}/{exp_config.epochs}\t l= {loss:.2f} \t phi= {phi:.2f}")
        )
        tracker.track(
            ev.TrackMetrics(
                "",
                {
                    "epoch.loss.predictor": float(loss),
                    "epoch.phi.predictor": float(phi),
                },
                epoch,
            )
        )

        sch_pred.step(loss)

        if (epoch + 1) % increase_after_epochs == 0:
            t = t * increase_rate
            tracker.track(ev.Log("", f"Increase t by {increase_rate} to {t}"))

    return (None, predictor)


def train_separate(
    models: Tuple[base.ConstrainedModule],
    loaders: Tuple[DataLoader],
    exp_config: BaseExperimentConfig,
    optimizers: List[Optimizer],
    schedulers: List[lr_scheduler.ReduceLROnPlateau],
    loss_function: nn.Module,
    tracker: AggregatedTracker = AggregatedTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    init_loader, pred_loader = loaders
    initializer, predictor = models
    opt_init, opt_pred = optimizers
    sch_init, sch_pred = schedulers
    initializer.initialize_parameters()
    initializer.set_lure_system()
    initializer.train()
    problem_status = predictor.initialize_parameters()
    tracker.track(ev.Log("", f"Initialize parameters: {problem_status}"))
    predictor.set_lure_system()
    predictor.train()
    t, increase_rate, increase_after_epochs = (
        exp_config.t,
        exp_config.increase_rate,
        exp_config.increase_after_epochs,
    )

    # initializer
    for epoch in range(exp_config.epochs):
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
            ev.Log("", f"{epoch}/{exp_config.epochs} (initializer)\t l= {loss:.2f}")
        )
        tracker.track(
            ev.TrackMetrics("", {"epoch.loss.initializer": float(loss)}, epoch)
        )

        sch_init.step(loss)

    # predictor
    for epoch in range(exp_config.epochs):
        loss, phi = 0.0, 0.0  # phi is barrier
        for step, batch in enumerate(pred_loader):
            predictor.zero_grad()
            initializer.zero_grad()
            with torch.no_grad():
                _, h0 = initializer.forward(batch["d_init"])
            e_hat, _ = predictor.forward(batch["d"], h0)
            batch_loss = loss_function(e_hat, batch["e"])
            batch_phi = predictor.get_phi(t)
            (batch_loss + batch_phi).backward()
            opt_pred.step()
            predictor.set_lure_system()
            loss += batch_loss.item()
            phi += batch_phi.item()

        if epoch % PLOT_AFTER_EPOCHS == 0:
            fig = plot.plot_sequence(
                [
                    e_hat[0, :].cpu().cpu().detach().numpy(),
                    batch["e"][0, :].cpu().detach().numpy(),
                ],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(ev.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            problem_status = predictor.project_parameters()
            tracker.track(ev.Log("", f"Projecting parameters: {problem_status}"))
            predictor.set_lure_system()

        tracker.track(
            ev.Log("", f"{epoch}/{exp_config.epochs}\t l= {loss:.2f} \t phi= {phi:.2f}")
        )
        tracker.track(
            ev.TrackMetrics(
                "",
                {
                    "epoch.loss.predictor": float(loss),
                    "epoch.phi.predictor": float(phi),
                },
                epoch,
            )
        )

        sch_pred.step(loss)

        if (epoch + 1) % increase_after_epochs == 0:
            t = t * increase_rate
            tracker.track(ev.Log("", f"Increase t by {increase_rate} to {t}"))

    return (initializer, predictor)


class Armijo:
    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        dF: Callable[[torch.Tensor], torch.Tensor],
        s0: float = 1.0,
        alpha: float = 0.4,
        beta: float = 0.4,
    ):
        self.f = f
        self.dF = dF
        self.s0 = s0
        self.alpha = alpha
        self.beta = beta

    def linesearch(self, theta: torch.Tensor, dir: torch.Tensor) -> torch.Tensor:
        i = 0
        f, dF, s, alpha, beta = self.f, self.dF, self.s0, self.alpha, self.beta
        while f(theta + s * dir) > f(theta) + alpha * s * dir.T @ dF(
            theta
        ) or torch.isnan(f(theta + s * dir)):
            s = beta * s
            i += 1
            if i > 100:
                raise ValueError
        # print(f'linesearch steps: {i}')
        return s

    def plot_step_size_function(self, theta) -> Figure:
        ss, s = [], self.s0
        for i in range(10):
            ss.append(s)
            s = self.beta * s

        fig, ax = plt.subplots(figsize=(10, 5))
        dir = -self.dF(theta)
        s = np.linspace(0, 1, 100)
        # print(f'dF(theta):{self.dF(theta)}, theta:{theta}')
        ax.plot(
            s,
            np.vectorize(lambda s: (self.f(theta + s * dir)).detach().numpy())(s),
            label="f(x+sd)",
        )
        ax.plot(
            s,
            np.vectorize(
                lambda s: (self.f(theta) + s * self.dF(theta).T @ dir).detach().numpy()
            )(s),
            label="f(x)+s dF.T d",
        )
        ax.plot(
            s,
            np.vectorize(
                lambda s: (self.f(theta) + self.alpha * s * self.dF(theta).T @ dir)
                .detach()
                .numpy()
            )(s),
            label="f(x)+alpha s dF.T d",
        )
        # for s in ss:
        #     ax.plot(s,self.f(theta)+self.alpha*s*self.dF(theta).T @ dir, 'x')
        #     ax.plot(s,self.f(theta)+s*self.dF(theta).T @ dir, 'x')
        #     ax.plot(s,self.f(theta+s*dir),'x')
        ax.grid()
        ax.set_xlabel("s")
        ax.legend()

        return fig
