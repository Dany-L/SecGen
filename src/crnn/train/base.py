import json
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from ..configuration.experiment import BaseExperimentConfig, load_configuration
from ..data_io import (get_result_directory_name, load_data,
                       load_initialization, load_normalization)
from ..datasets import get_datasets, get_loaders
from ..loss import get_loss_function
from ..models import base
from ..models.model_io import (get_model_from_config, load_model,
                               set_parameters_to_train)
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker
from ..tracker.tracker_io import get_trackers_from_config
from ..utils import base as utils

PLOT_AFTER_EPOCHS: int = 100


class InitPred(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(
        self,
        models: List[base.DynamicIdentificationModel],
        loaders: List[DataLoader],
        exp_config: BaseExperimentConfig,
        optimizers: List[Optimizer],
        loss_function: nn.Module,
        schedulers: List[lr_scheduler.ReduceLROnPlateau],
        tracker: AggregatedTracker = AggregatedTracker(),
        initialize: bool = True,
    ) -> Tuple[Optional[base.DynamicIdentificationModel]]:
        pass


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

    # training data
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
    # validation data
    val_inputs, val_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        "validation",
        dataset_dir,
    )
    n_val_inputs = utils.normalize(val_inputs, input_mean, input_std)
    n_val_outputs = utils.normalize(val_outputs, output_mean, output_std)

    start_time = time.time()
    with torch.device(device):
        loaders = [
            get_loaders(
                get_datasets(
                    inp,
                    output,
                    horizon,
                    experiment_config.window,
                ),
                batch_size,
                device,
            )
            for inp, output, horizon, batch_size in zip(
                [n_train_inputs, n_val_inputs],
                [n_train_outputs, n_val_outputs],
                [
                    experiment_config.horizons.training,
                    experiment_config.horizons.validation,
                ],
                [experiment_config.batch_size, 1],
            )
        ]

        init_data = load_initialization(result_directory)
        (initializer, predictor) = get_model_from_config(model_config)

        initializer.set_lure_system()
        init_data = predictor.initialize_parameters(
            n_train_inputs, n_train_outputs, init_data.data
        )
        tracker.track(ev.Log("", init_data.msg))
        if init_data.data:
            tracker.track(ev.SaveInitialization("", init_data.data["ss"], {}))
        optimizers = get_optimizer(experiment_config, (initializer, predictor))
        schedulers = get_scheduler(optimizers)

        loss_fcn = get_loss_function(experiment_config.loss_function)

        initial_hidden_state = experiment_config.initial_hidden_state
        trainer_class = retrieve_trainer_class(
            f"crnn.train.{initial_hidden_state}.{initial_hidden_state.title()}InitPredictor"
        )
        trainer = trainer_class()

        (initializer, predictor) = trainer.train(
            models=(initializer, predictor),
            loaders=loaders,
            exp_config=experiment_config,
            optimizers=optimizers,
            loss_function=loss_fcn,
            schedulers=schedulers,
            tracker=tracker,
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


def continue_training(
    config_file_name: str,
    dataset_dir: str,
    result_base_directory: str,
    model_name: str,
    experiment_name: str,
    gpu: bool,
) -> None:
    device = utils.get_device(gpu)

    result_directory = get_result_directory_name(
        result_base_directory, model_name, experiment_name
    )
    config = load_configuration(config_file_name)
    trackers_config = config.trackers
    experiment_config = config.experiments[experiment_name]
    model_name = f"{experiment_name}-{model_name}"
    model_config = config.models[model_name]

    trackers = get_trackers_from_config(
        trackers_config, result_directory, model_name, "training"
    )
    tracker = AggregatedTracker(trackers)
    tracker.track(ev.LoadTrackingConfiguration("", result_directory, model_name))
    tracker.track(ev.Log("", "Continue training"))

    train_inputs, train_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        "train",
        dataset_dir,
    )
    normalization = load_normalization(result_directory)

    n_train_inputs = utils.normalize(
        train_inputs, normalization.input.mean, normalization.input.std
    )
    n_train_outputs = utils.normalize(
        train_outputs, normalization.output.mean, normalization.output.std
    )

    # validation data
    val_inputs, val_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        "validation",
        dataset_dir,
    )
    n_val_inputs = utils.normalize(
        val_inputs, normalization.input.mean, normalization.input.std
    )
    n_val_outputs = utils.normalize(
        val_outputs, normalization.output.mean, normalization.output.std
    )

    start_time = time.time()
    with torch.device(device):
        loaders = [
            get_loaders(
                get_datasets(
                    inp,
                    output,
                    horizon,
                    experiment_config.window,
                ),
                batch_size,
                device,
            )
            for inp, output, horizon, batch_size in zip(
                [n_train_inputs, n_val_inputs],
                [n_train_outputs, n_val_outputs],
                [
                    experiment_config.horizons.training,
                    experiment_config.horizons.validation,
                ],
                [experiment_config.batch_size, 1],
            )
        ]

        initializer, predictor = get_model_from_config(model_config)

        optimizers = get_optimizer(experiment_config, (initializer, predictor))
        schedulers = get_scheduler(optimizers)
        initializer, predictor = load_model(
            experiment_config, initializer, predictor, model_name, result_directory
        )

        loss_fcn = get_loss_function(experiment_config.loss_function)

        initial_hidden_state = experiment_config.initial_hidden_state
        trainer_class = retrieve_trainer_class(
            f"crnn.train.{initial_hidden_state}.{initial_hidden_state.title()}InitPredictor"
        )
        trainer = trainer_class()

        # load trainable parameters
        trained_parameters_filename = os.path.join(
            result_directory, f"model_params-{model_name}.json"
        )
        if os.path.exists(trained_parameters_filename):
            with open(trained_parameters_filename, mode="r") as f:
                par_dict = json.load(f)

            all_pars, trained_pars = set(par_dict["all_parameters"]), set(
                par_dict["trained_parameters"]
            )
            if not all_pars == trained_pars:
                tracker.track(
                    ev.Log("", f"Parameters to train: {all_pars-trained_pars}")
                )
                for model in [initializer, predictor]:
                    set_parameters_to_train(model, par_dict["trained_parameters"])

        else:
            raise NameError(
                f"The parameters are not trained, first train the model {model_name}."
            )

        # if not predictor.check_constraints():
        #     tracker.track(
        #         ev.Log(
        #             "", "Loaded parameters are not feasible, project onto feasible set."
        #         )
        #     )
        #     result = predictor.project_parameters()
        #     tracker.track(ev.Log("", f"Projection result: {result}"))

        (initializer, predictor) = trainer.train(
            models=(initializer, predictor),
            loaders=loaders,
            exp_config=experiment_config,
            optimizers=optimizers,
            loss_function=loss_fcn,
            schedulers=schedulers,
            tracker=tracker,
            initialize=False,
        )

    stop_time = time.time()
    training_duration = utils.get_duration_str(start_time, stop_time)
    tracker.track(ev.SaveModel("", predictor, "predictor"))
    if initializer is not None:
        tracker.track(ev.SaveModel("", initializer, "initializer"))
    tracker.track(ev.SaveModelParameter("", predictor,'-update'))
    tracker.track(ev.Log("", f"Training duration: {training_duration}"))
    tracker.track(ev.TrackMetrics("", {"training_duration": stop_time - start_time}))
    tracker.track(ev.Stop(""))


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

    # @partial(jit, static_argnames=['self'])
    def linesearch(self, theta: torch.Tensor, dir: torch.Tensor) -> torch.Tensor:
        f, dF, s, alpha, beta = self.f, self.dF, self.s0, self.alpha, self.beta
        i = 0
        while f(theta + s * dir) > f(theta) + alpha * s * dir.T @ dF(
            theta
        ) or self.get_isnan(f(theta + s * dir)):

            s = beta * s
            # print(torch.squeeze(theta+s*dir))
            i += 1
            if i > 100:
                # raise ValueError
                return s
        # print(f'linesearch steps: {i}')
        return s

    def get_isnan(self, value):
        if isinstance(value, torch.Tensor):
            return torch.isnan(value)
        else:
            return np.isnan(value)

    def plot_step_size_function(self, theta) -> Figure:
        ss, s = [], self.s0
        for i in range(10):
            ss.append(s)
            s = self.beta * s

        fig, ax = plt.subplots(figsize=(10, 5))
        dir = -self.dF(theta)
        s = np.linspace(0, 1, 100)
        # print(f'dF(theta):{self.dF(theta)}, theta:{theta}')
        if isinstance(theta, torch.Tensor):
            ax.plot(
                s,
                np.vectorize(lambda s: (self.f(theta + s * dir)).detach().numpy())(s),
                label="f(x+sd)",
            )
            ax.plot(
                s,
                np.vectorize(
                    lambda s: (self.f(theta) + s * self.dF(theta).T @ dir)
                    .detach()
                    .numpy()
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
        else:
            ax.plot(
                s,
                np.vectorize(lambda s: (self.f(theta + s * dir)))(s),
                label="f(x+sd)",
            )
            f_s_dF_d = np.zeros_like(s)
            for idx, s_i in enumerate(s):
                f_s_dF_d[idx] = np.squeeze(self.f(theta) + s_i * self.dF(theta).T @ dir)

            ax.plot(
                s,
                f_s_dF_d,
                label="f(x)+s dF.T d",
            )
            f_a_s_a_dF_d = np.zeros_like(s)
            for idx, s_i in enumerate(s):
                f_a_s_a_dF_d[idx] = np.squeeze(
                    self.f(theta) + self.alpha * s_i * self.dF(theta).T @ dir
                )
            ax.plot(
                s,
                f_a_s_a_dF_d,
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


def retrieve_trainer_class(
    model_class_string: str,
) -> Type[InitPred]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split(".")
    module_string = ".".join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, InitPred):
        raise ValueError(f"{cls} is not a subclass of InitPred")
    return cls  # type: ignore
