from typing import Tuple

import numpy as np
import torch

from . import metrics
from . import tracker as base_tracker
from .configuration import Normalization
from .datasets import RecurrentWindowHorizonDataset
from .models import base
from .utils import base as utils
from .utils import plot


def evaluate_model(
    models: Tuple[base.ConstrainedModule],
    test_dataset: RecurrentWindowHorizonDataset,
    metrics: metrics.Metrics,
    normalization: Normalization,
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
    e = metrics.forward(es, e_hats)
    num_par = sum(p.numel() for p in predictor.parameters() if p.requires_grad)

    tracker.track(base_tracker.SaveSequences("", e_hats, es, "test_output"))
    tracker.track(base_tracker.Log("", f"RMSE: {np.mean(e):.2f}"))
    tracker.track(base_tracker.Log("", f"Number of parameters: {num_par}"))

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
