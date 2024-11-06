from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from . import tracker as base_tracker
from .models import base
from .utils import plot


def train_joint(
    models: Tuple[base.ConstrainedModule],
    train_loader: DataLoader,
    epochs: int,
    optimizers: Tuple[Optimizer],
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    initializer, predictor = models
    opt_init, opt_pred = optimizers
    predictor.initialize_parameters()
    predictor.set_lure_system()
    predictor.train()
    tracker.track(base_tracker.Start(""))

    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            initializer.zero_grad()
            e_hat_init, h0 = initializer.forward(batch["d_init"])
            e_hat, _ = predictor.forward(batch["d"], h0)
            batch_loss_predictor = loss_function(e_hat, batch["e"])
            batch_loss_initializer = loss_function(
                e_hat_init[:, -1, :], batch["e_init"][:, -1, :]
            )
            batch_loss = batch_loss_predictor + batch_loss_initializer
            batch_loss.backward()
            opt_pred.step()
            predictor.set_lure_system()
            loss += batch_loss.item()
        if epoch % 20 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            predictor.project_parameters()
            predictor.set_lure_system()
        tracker.track(base_tracker.Log("", f"{epoch}/{epochs}\t l= {loss:.2f}"))
    tracker.track(base_tracker.Stop(""))

    return (initializer, predictor)


def train_zero(
    models: Tuple[base.ConstrainedModule],
    train_loader: DataLoader,
    epochs: int,
    optimizers: Tuple[Optimizer],
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    _, predictor = models
    _, opt_pred = optimizers
    predictor.initialize_parameters()
    predictor.set_lure_system()
    predictor.train()
    tracker.track(base_tracker.Start(""))

    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            e_hat, _ = predictor.forward(batch["d"])
            batch_loss = loss_function(e_hat, batch["e"])
            batch_loss.backward()
            opt_pred.step()
            predictor.set_lure_system()
            loss += batch_loss.item()
        if epoch % 20 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            predictor.project_parameters()
            predictor.set_lure_system()
        tracker.track(base_tracker.Log("", f"{epoch}/{epochs}\t l= {loss:.2f}"))
    tracker.track(base_tracker.Stop(""))

    return (None, predictor)


def train_separate(
    models: Tuple[base.ConstrainedModule],
    train_loader: DataLoader,
    epochs: int,
    optimizers: Tuple[Optimizer],
    loss_function: nn.Module,
    tracker: base_tracker.BaseTracker = base_tracker.BaseTracker(),
) -> Tuple[Optional[base.ConstrainedModule]]:
    initializer, predictor = models
    opt_init, opt_pred = optimizers
    predictor.initialize_parameters()
    predictor.set_lure_system()
    predictor.train()
    tracker.track(base_tracker.Start(""))

    # initializer
    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            initializer.zero_grad()
            e_hat_init, _ = initializer.forward(batch["d_init"])
            batch_loss = loss_function(e_hat_init[:, -1, :], batch["e_init"][:, -1, :])
            batch_loss.backward()
            opt_init.step()
            loss += batch_loss.item()

        tracker.track(
            base_tracker.Log("", f"{epoch}/{epochs} (initializer)\t l= {loss:.2f}")
        )

    # predictor
    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            predictor.zero_grad()
            _, h0 = initializer.forward(batch["d_init"])
            e_hat, _ = predictor.forward(batch["d"], h0)
            batch_loss = loss_function(e_hat, batch["e"])
            batch_loss.backward()
            opt_pred.step()
            predictor.set_lure_system()
            loss += batch_loss.item()

        if epoch % 20 == 0:
            fig = plot.plot_sequence(
                [e_hat[0, :].detach().numpy(), batch["e"][0, :].detach().numpy()],
                0.01,
                title="normalized output",
                legend=[r"$\hat e$", r"$e$"],
            )
            tracker.track(base_tracker.SaveFig("", fig, f"e_{epoch}"))

        if not predictor.check_constraints():
            predictor.project_parameters()
            predictor.set_lure_system()

        tracker.track(
            base_tracker.Log("", f"{epoch}/{epochs} (predictor)\t l= {loss:.2f}")
        )
    tracker.track(base_tracker.Stop(""))

    return (initializer, predictor)
