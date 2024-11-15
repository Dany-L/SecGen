import os
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch

from . import metrics
from .additional_tests import (AdditionalTest, AdditionalTestResult,
                               retrieve_additional_test_class)
from .configuration.base import InputOutput, Normalization
from .configuration.experiment import load_configuration
from .data_io import get_result_directory_name, load_data, load_normalization
from .datasets import RecurrentWindowHorizonDataset
from .metrics import retrieve_metric_class
from .models import base as base_models
from .models.model_io import get_model_from_config
from .tracker import events as ev
from .tracker.base import AggregatedTracker, get_trackers_from_config
from .utils import base as utils
from .utils import plot


def evaluate(
    config_file_name: str,
    dataset_dir: str,
    result_base_directory: str,
    model_name: str,
    experiment_name: str,
) -> None:
    mode = "test"

    result_directory = get_result_directory_name(
        result_base_directory, model_name, experiment_name
    )
    config = load_configuration(config_file_name)
    trackers_config = config.trackers
    experiment_config = config.experiments[experiment_name]
    model_name = f"{experiment_name}-{model_name}"
    model_config = config.models[model_name]
    metrics_config = config.metrics
    additional_tests_config = config.additional_tests

    dataset_name = os.path.basename(os.path.dirname(dataset_dir))

    trackers = get_trackers_from_config(
        trackers_config, result_directory, model_name, "validation"
    )
    tracker = AggregatedTracker(trackers)
    tracker.track(ev.LoadTrackingConfiguration("", result_directory, model_name))

    test_inputs, test_outputs = load_data(
        experiment_config.input_names, experiment_config.output_names, mode, dataset_dir
    )
    normalization = load_normalization(result_directory)
    n_test_inputs = utils.normalize(
        test_inputs, normalization.input.mean, normalization.input.std
    )

    test_dataset = RecurrentWindowHorizonDataset(
        n_test_inputs,
        test_outputs,
        experiment_config.horizons.testing,
        experiment_config.window,
    )

    initializer, predictor = get_model_from_config(model_config)

    if experiment_config.initial_hidden_state == "zero":
        initializer, predictor = None, base_models.load_model(
            predictor,
            os.path.join(
                result_directory, utils.get_model_file_name("predictor", model_name)
            ),
        )
    else:
        initializer, predictor = (
            base_models.load_model(
                model,
                os.path.join(
                    result_directory, utils.get_model_file_name(name, model_name)
                ),
            )
            for model, name in zip(
                [initializer, predictor], ["initializer", "predictor"]
            )
        )

    metrics = dict()
    for name, metric_config in metrics_config.items():
        metric_class = retrieve_metric_class(metric_config.metric_class)
        metrics[name] = metric_class(metric_config.parameters)

    additional_tests = dict()
    for name, additional_test_config in additional_tests_config.items():
        additional_test_class = retrieve_additional_test_class(
            additional_test_config.test_class
        )
        additional_tests[name] = additional_test_class(
            additional_test_config.parameters, predictor, tracker
        )

    evaluate_model(
        (initializer, predictor),
        test_dataset,
        metrics,
        normalization,
        mode,
        additional_tests,
        dataset_name,
        tracker,
    )
    tracker.track(ev.Stop(""))


def evaluate_model(
    models: Tuple[base_models.ConstrainedModule],
    test_dataset: RecurrentWindowHorizonDataset,
    metrics: Dict[str, metrics.Metrics],
    normalization: Normalization,
    mode: Literal["test", "validation"],
    additional_tests: Dict[str, AdditionalTest],
    dataset_name: str,
    tracker: AggregatedTracker = AggregatedTracker(),
) -> None:
    initializer, predictor = models

    tracker.track(ev.Start("", dataset_name))
    tracker.track(ev.Log("", f"Constraints satisfied? {predictor.check_constraints()}"))

    es, e_hats, ds = [], [], []
    for _, sample in enumerate(test_dataset):
        if initializer is not None:
            _, h0 = initializer.forward(
                torch.unsqueeze(torch.tensor(sample["d_init"]), 0)
            )
        else:
            h0 = None
        e_hat, _ = predictor.forward(torch.unsqueeze(torch.tensor(sample["d"]), 0), h0)
        e_hats.append(torch.squeeze(e_hat, 0).detach().numpy())
        es.append(sample["e"])
        ds.append(sample["d"])

    e_hats = utils.denormalize(
        e_hats, normalization.output.mean, normalization.output.std
    )
    ds = utils.denormalize(ds, normalization.input.mean, normalization.input.std)
    input_outputs = [
        InputOutput(d=d, e_hat=e_hat, e=e) for d, e_hat, e in zip(ds, e_hats, es)
    ]

    results: Dict[str, Any] = dict()
    metrics_result: Dict[str, float] = dict()
    for name, metric in metrics.items():
        e = metric.forward(es, e_hats)
        tracker.track(ev.Log("", f"{name}: {np.mean(e):.2f}"))
        tracker.track(ev.TrackMetrics("", {name: float(np.mean(e))}))
        metrics_result[name] = float(e)
    results["num_parameters"] = sum(
        p.numel() for p in predictor.parameters() if p.requires_grad
    )
    results["metrics"] = metrics_result

    additional_test_results: Dict[str, AdditionalTestResult] = dict()
    for name, additional_test in additional_tests.items():
        tracker.track(ev.Log("", f"Running additional test {name}"))
        result = additional_test.test()
        tracker.track(ev.Log("", f"{name}: {result.value:.2f}"))
        tracker.track(
            ev.SaveFig(
                "",
                plot.plot_sequence(
                    [result.input_output[0].e_hat], 0.01, name, ["e_hat"]
                ),
                f"{name}-{dataset_name}",
            )
        )
        additional_test_results[name] = dict(
            value=result.value,
            additional=result.additional,
        )
        tracker.track(
            ev.SaveSequences(
                "",
                result.input_output,
                f"test_output-{name}-{dataset_name}",
            )
        )
    results["additional_tests"] = additional_test_results

    tracker.track(ev.SaveSequences("", input_outputs, f"test_output-{dataset_name}"))
    tracker.track(ev.Log("", f"Number of parameters: {results['num_parameters']}"))

    tracker.track(ev.TrackParameters("", f"{mode}-eval", results))

    tracker.track(
        ev.SaveFig(
            "",
            plot.plot_sequence(
                [es[0], e_hats[0]], 0.01, f"RMSE {np.mean(e):.2f}", ["e", "e_hat"]
            ),
            "test_output",
        )
    )
