from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from . import tracker as base_tracker
from .models import base
from .utils import plot


def train_model(
    model: base.ConstrainedModule,
    train_loader: DataLoader,
    epochs: int,
    optimizer: Optimizer,
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> base.ConstrainedModule:

    model.initialize_parameters()
    model.set_lure_system()
    model.train()
    tracker.track(base_tracker.Start(""))

    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            model.zero_grad()
            e_hat, _ = model.forward(batch["d"])
            batch_loss = loss_function(e_hat, batch["e"])
            batch_loss.backward()
            optimizer.step()
            model.set_lure_system()
            loss += batch_loss
        if epoch % 100 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not model.check_constraints():
            model.project_parameters()
            model.set_lure_system()
        tracker.track(base_tracker.Log("", f"{epoch}/{epochs}\t l= {loss:.2f}"))
    tracker.track(base_tracker.Stop(""))

    return model
