import json
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from ..configuration.experiment import BaseExperimentConfig, load_configuration
from ..configuration.base import PreprocessingResults, Normalization, PREPARED_FOLDER_NAME, TRAIN_FOLDER_NAME, VALIDATION_FOLDER_NAME, IN_DISTRIBUTION_FOLDER_NAME, OUT_OF_DISTRIBUTION_FOLDER_NAME, SystemIdentificationResults
from ..io.data import (
    get_result_directory_name,
    load_data,
    load_initialization,
    load_normalization,
    check_data_availability,
    load_preprocessing_results,
)
from ..datasets import get_datasets, get_loaders
from ..loss import get_loss_function
from ..models import base
from ..io.model import get_model_from_config, load_model, set_parameters_to_train
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker
from ..io.tracker import get_trackers_from_config
from ..utils import base as utils
from ..systemtheory.analysis import get_transient_time

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
        raw_data: Optional[Dict[str, Any]] = None,
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
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))
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

    # Check if preprocessing results are available
    preprocessing_results = load_preprocessing_results(
        result_directory, experiment_config, dataset_dir
    )

    if preprocessing_results is None:
        tracker.track(
            ev.Log(
                "", "No preprocessing results found. Please run preprocessing.py first."
            )
        )
        raise RuntimeError(
            f"Preprocessing results not found in {result_directory}. "
            "Please run scripts/preprocessing.py before training."
        )

    # Check data availability
    data_availability = check_data_availability(dataset_dir)
    missing_data = [
        data_type for data_type, available in data_availability.items() if not available
    ]

    if missing_data:
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))
        tracker.track(ev.Log("", f"Missing data types: {missing_data}"))
        raise RuntimeError(
            f"Missing data in {dataset_dir}: {missing_data}. "
            f"Please run scripts/prepare_data.py {dataset_name} {dataset_dir} first."
        )

    # Extract preprocessing results
    ss = preprocessing_results.system_identification.ss
    transient_time = preprocessing_results.system_identification.transient_time
    horizon = preprocessing_results.horizon
    window = preprocessing_results.window

    input_mean = preprocessing_results.normalization.input.mean
    input_std = preprocessing_results.normalization.input.std
    output_mean = preprocessing_results.normalization.output.mean
    output_std = preprocessing_results.normalization.output.std

    tracker.track(
        ev.Log(
            "", f"Using preprocessing results - window: {window}, horizon: {horizon}"
        )
    )
    tracker.track(ev.TrackParameter("", "transient_time", transient_time))
    processed_directory = os.path.join(dataset_dir, PREPARED_FOLDER_NAME)

    # Load and normalize data
    train_inputs, train_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        TRAIN_FOLDER_NAME,
        processed_directory,
    )
    tracker.track(
        ev.Log(
            "",
            f"Training samples: {np.sum([train_input.shape[0] for train_input in train_inputs])}",
        )
    )

    val_inputs, val_outputs = load_data(
        experiment_config.input_names,
        experiment_config.output_names,
        VALIDATION_FOLDER_NAME,
        processed_directory,
    )
    tracker.track(
        ev.Log(
            "",
            f"Validation samples: {np.sum([val_input.shape[0] for val_input in val_inputs])}",
        )
    )

    # Normalize data using preprocessing parameters
    n_train_inputs = utils.normalize(train_inputs, input_mean, input_std)
    n_train_outputs = utils.normalize(train_outputs, output_mean, output_std)
    n_val_inputs = utils.normalize(val_inputs, input_mean, input_std)
    n_val_outputs = utils.normalize(val_outputs, output_mean, output_std)

    # Create init_data for model initialization
    init_data = load_initialization(result_directory)

    start_time = time.time()
    with torch.device(device):
        loaders = [
            get_loaders(
                get_datasets(
                    inp,
                    output,
                    h,
                    window,
                ),
                batch_size,
                device,
            )
            for inp, output, h, batch_size in zip(
                [n_train_inputs, n_val_inputs],
                [n_train_outputs, n_val_outputs],
                [horizon, horizon],
                [experiment_config.batch_size, 1],
            )
        ]

        (initializer, predictor) = get_model_from_config(model_config)

        initializer.set_lure_system()
        # predictor.set_lure_system()
        init_data = predictor.initialize_parameters(
            n_train_inputs, n_train_outputs, init_data.data
        )
        tracker.track(ev.Log("", init_data.msg))
        # if init_data.data:
        #     tracker.track(ev.SaveInitialization("", init_data.data["ss"], {}))
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
                    ev.Log("", f"Parameters to train: {all_pars - trained_pars}")
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
    tracker.track(ev.SaveModelParameter("", predictor, "-update"))
    tracker.track(ev.Log("", f"Training duration: {training_duration}"))
    tracker.track(ev.TrackMetrics("", {"training_duration": stop_time - start_time}))
    tracker.track(ev.Stop(""))


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
