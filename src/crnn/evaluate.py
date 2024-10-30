import numpy as np
import torch

from . import metrics
from . import tracker as base_tracker
from .datasets import RecurrentWindowHorizonDataset
from .models import base
from .utils import plot


def evaluate_model(
    model: base.ConstrainedModule,
    test_dataset: RecurrentWindowHorizonDataset,
    metrics: metrics.Metrics,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> None:

    tracker.track(base_tracker.Start(""))

    es, e_hats = [], []
    for _, sample in enumerate(test_dataset):
        e_hat, _ = model.forward(torch.unsqueeze(torch.tensor(sample["d"]), 0))
        e_hats.append(torch.squeeze(e_hat, 0).detach().numpy())
        es.append(sample["e"])

    e = metrics.forward(es, e_hats)

    tracker.track(base_tracker.SaveSequences("", e_hats, es, "test_output"))
    tracker.track(base_tracker.Log("", f"RMSE: {np.mean(e):.2f}"))
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
