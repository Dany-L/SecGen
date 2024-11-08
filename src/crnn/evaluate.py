from typing import Tuple, Dict, Literal

import numpy as np
import torch

from . import metrics
from . import tracker as base_tracker
from .configuration import Normalization
from .utils import plot

from  . import tracker as base_tracker
from . import configuration as cfg
from .data_io import (get_result_directory_name, load_data,
                          load_normalization)
from .datasets import RecurrentWindowHorizonDataset
from .models import base as base_models
from .models.model_io import get_model_from_config
from .utils import base as utils
from .metrics import retrieve_metric_class
import os


def evaluate(
    config_file_name: str,
    dataset_dir: str,
    result_base_directory: str,
    model_name: str,
    experiment_name: str,
) -> None:
    mode = "test"

    result_directory = get_result_directory_name(result_base_directory, model_name, experiment_name)
    tracker = base_tracker.BaseTracker(result_directory, model_name, "validation")
    config = cfg.load_configuration(config_file_name)
    experiment_config = config.experiments[experiment_name]
    model_config = config.models[model_name]
    metrics_config = config.metrics

    test_inputs, test_outputs = load_data(
        experiment_config.input_names, experiment_config.output_names, mode, dataset_dir
    )
    normalization = load_normalization(result_directory)
    n_test_inputs = utils.normalize(
        test_inputs, normalization.input.mean, normalization.input.std
    )

    test_dataset = RecurrentWindowHorizonDataset(
        n_test_inputs, test_outputs, experiment_config.horizons.testing, experiment_config.window
    )

    initializer, predictor = get_model_from_config(model_config)

    if experiment_config.initial_hidden_state == "zero":
        initializer, predictor = None, base_models.load_model(predictor, os.path.join(result_directory, utils.get_model_file_name('predictor',model_name)))
    else:
        initializer, predictor = (base_models.load_model(
            model,
            os.path.join(result_directory, utils.get_model_file_name(name,model_name)),
        ) for model, name in zip([initializer, predictor],['initializer', 'predictor']))
    
    metrics = dict()
    for name, metric_config in metrics_config.items():
        metric_class = retrieve_metric_class(metric_config.metric_class)
        metrics[name] = metric_class(metric_config.parameters)

    evaluate_model((initializer,predictor), test_dataset, metrics, normalization, mode, tracker)

def evaluate_model(
    models: Tuple[base_models.ConstrainedModule],
    test_dataset: RecurrentWindowHorizonDataset,
    metrics: Dict[str, metrics.Metrics],
    normalization: Normalization,
    mode: Literal["test", "validation"],
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> None:
    initializer, predictor = models

    tracker.track(base_tracker.Start(""))
    tracker.track(
        base_tracker.Log("", f"Constraints satisfied? {predictor.check_constraints()}")
    )

    es, e_hats = [], []
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

    e_hats = utils.denormalize(
        e_hats, normalization.output.mean, normalization.output.std
    )

    results: Dict[str, float] = dict()
    for name, metric in metrics.items():
        e = metric.forward(es, e_hats)
        tracker.track(base_tracker.Log("", f"{name}: {np.mean(e):.2f}"))
        results[name] = float(e)
    results["num_parameters"] = sum(p.numel() for p in predictor.parameters() if p.requires_grad)

    tracker.track(base_tracker.SaveSequences("", e_hats, es, "test_output"))
    tracker.track(base_tracker.Log("", f"Number of parameters: {results['num_parameters']}"))
    tracker.track(base_tracker.SaveEvaluation("", results, mode))
    

    tracker.track(
        base_tracker.SaveFig(
            "",
            plot.plot_sequence(
                [es[0], e_hats[0]], 0.01, f"RMSE {np.mean(e):.2f}", ["e", "e_hat"]
            ),
            "test_output",
        )
    )
    tracker.track(base_tracker.Stop(""))
