import os
import time
from abc import abstractmethod
from typing import Callable, Type, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from ..configuration.experiment import load_configuration, BaseExperimentConfig
from ..data_io import get_result_directory_name, load_data
from ..datasets import get_datasets, get_loaders
from ..loss import get_loss_function
from ..models.model_io import get_model_from_config
from ..models import base
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker
from ..tracker.tracker_io import get_trackers_from_config
from ..utils import base as utils


PLOT_AFTER_EPOCHS: int = 100


class InitPred:

    @abstractmethod
    def train(
        models: Tuple[base.DynamicIdentificationModel],
        loaders: Tuple[DataLoader],
        exp_config: BaseExperimentConfig,
        optimizers: List[Optimizer],
        loss_function: nn.Module,
        schedulers: List[lr_scheduler.ReduceLROnPlateau],
        tracker: AggregatedTracker = AggregatedTracker(),
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
        optimizers = get_optimizer(experiment_config, (initializer, predictor))
        schedulers = get_scheduler(optimizers)

        loss_fcn = get_loss_function(experiment_config.loss_function)

        initial_hidden_state = experiment_config.initial_hidden_state
        trainer = retrieve_trainer_class(
            f"crnn.train.{initial_hidden_state}.{initial_hidden_state.title()}InitPredictor"
        )

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
            i += 1
            if i > 100:
                raise ValueError
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