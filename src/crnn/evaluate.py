import os
import time
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch

from . import metrics
from .additional_tests import (
    AdditionalTest,
    AdditionalTestResult,
    retrieve_additional_test_class,
)
from .configuration.base import InputOutput, Normalization
from .configuration.experiment import load_configuration
from .io.data import get_result_directory_name, load_data, load_normalization
from .datasets import RecurrentWindowHorizonDataset
from .metrics import retrieve_metric_class
from .models import base
from .io.model import get_model_from_config, load_model
from .tracker import events as ev
from .tracker.base import AggregatedTracker
from .io.tracker import get_trackers_from_config
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

    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))

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
    initializer, predictor = load_model(
        experiment_config, initializer, predictor, model_name, result_directory
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

    start_time = time.time()
    evaluate_model(
        (initializer, predictor),
        test_dataset,
        metrics,
        normalization,
        mode,
        additional_tests,
        dataset_name,
        experiment_config.dt,
        tracker,
    )
    stop_time = time.time()
    training_duration = utils.get_duration_str(start_time, stop_time)
    tracker.track(ev.Log("", f"Evaluation duration: {training_duration}"))
    tracker.track(ev.TrackMetrics("", {"evaluation_duration": stop_time - start_time}))
    tracker.track(ev.Stop(""))


def evaluate_model(
    models: Tuple[base.DynamicIdentificationModel],
    test_dataset: RecurrentWindowHorizonDataset,
    metrics: Dict[str, metrics.Metrics],
    normalization: Normalization,
    mode: Literal["test", "validation"],
    additional_tests: Dict[str, AdditionalTest],
    dataset_name: str,
    dt: float,
    tracker: AggregatedTracker = AggregatedTracker(),
) -> None:
    initializer, predictor = models

    tracker.track(ev.Start("", dataset_name))
    tracker.track(ev.Log("", f"Constraints satisfied? {predictor.check_constraints()}"))

    es, e_hats, ds = [], [], []
    for _, sample in enumerate(test_dataset):
        if hasattr(predictor, "theta"):
            theta = predictor.theta
        else:
            theta = None
        if initializer is not None and isinstance(
            initializer, base.DynamicIdentificationModel
        ):
            _, h0 = initializer.forward(
                torch.unsqueeze(torch.tensor(sample["d_init"]), 0)
            )
        else:
            h0 = None
        e_hat, _ = predictor.forward(
            torch.unsqueeze(torch.tensor(sample["d"]), 0), h0, theta
        )
        e_hats.append(torch.squeeze(e_hat, 0).cpu().detach().numpy())
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
        metrics_result[name] = float(np.mean(e))
    results["num_parameters"] = int(predictor.get_number_of_parameters())
    results["metrics"] = metrics_result

    additional_test_results: Dict[str, AdditionalTestResult] = dict()
    if isinstance(predictor, base.DynamicIdentificationModel):
        for name, additional_test in additional_tests.items():
            tracker.track(ev.Log("", f"Running additional test {name}"))
            result = additional_test.test()
            tracker.track(ev.Log("", f"{name}: {result.value:.2f}"))
            tracker.track(
                ev.SaveFig(
                    "",
                    plot.plot_sequence(
                        [result.input_output[0].e_hat[0]], 0.01, name, ["e_hat"]
                    ),
                    f"{name}-{dataset_name}",
                )
            )
            additional_test_results[name] = dict(value=result.value)
            tracker.track(
                ev.SaveSequences(
                    "",
                    result.input_output,
                    f"test_output-{name}-{dataset_name}",
                )
            )
            tracker.track(ev.TrackMetrics("", {name: result.value}))
    results["additional_tests"] = additional_test_results

    tracker.track(ev.SaveSequences("", input_outputs, f"test_output-{dataset_name}"))
    tracker.track(ev.Log("", f"Number of parameters: {results['num_parameters']}"))

    tracker.track(ev.TrackResults("", f"{mode}-eval", results))

    tracker.track(
        ev.SaveFig(
            "",
            plot.plot_sequence(
                [es[0], e_hats[0]], dt, f"RMSE {np.mean(e):.2f}", ["e", "e_hat"]
            ),
            "test_output",
        )
    )
