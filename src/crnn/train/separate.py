from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from ..configuration.experiment import BaseExperimentConfig
from ..models import base
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker
from ..utils import plot
from .base import PLOT_AFTER_EPOCHS, InitPred


class SeparateInitPredictor(InitPred):

    def train(
        self,
        models: Tuple[base.DynamicIdentificationModel],
        loaders: Tuple[DataLoader],
        exp_config: BaseExperimentConfig,
        optimizers: List[Optimizer],
        schedulers: List[lr_scheduler.ReduceLROnPlateau],
        loss_function: nn.Module,
        tracker: AggregatedTracker = AggregatedTracker(),
        initialize: bool = True,
    ) -> Tuple[Optional[base.DynamicIdentificationModel]]:
        train_loaders, validation_loaders = loaders
        init_loader, pred_loader = train_loaders
        _, validation_loader = validation_loaders
        initializer, predictor = models
        opt_init, opt_pred = optimizers
        sch_init, sch_pred = schedulers
        initializer.train()
        predictor.train()
        t, increase_rate, increase_after_epochs = (
            exp_config.t,
            exp_config.increase_rate,
            exp_config.increase_after_epochs,
        )

        # initializer
        for epoch in range(exp_config.epochs):
            loss = 0
            for step, batch in enumerate(init_loader):
                initializer.zero_grad()
                e_hat_init, _ = initializer.forward(batch["d"])
                batch_loss = loss_function(e_hat_init[:, -1, :], batch["e"][:, -1, :])
                batch_loss.backward()
                opt_init.step()
                initializer.set_lure_system()
                loss += batch_loss.item()

            tracker.track(
                ev.Log("", f"{epoch}/{exp_config.epochs} (initializer)\t l= {loss:.2f}")
            )
            tracker.track(
                ev.TrackMetrics("", {"epoch.loss.initializer": float(loss)}, epoch)
            )

            if sch_init is not None:
                sch_init.step(loss)

        # predictor
        for epoch in range(exp_config.epochs):
            loss, phi = 0.0, 0.0  # phi is barrier
            for step, batch in enumerate(pred_loader):
                predictor.zero_grad()
                initializer.zero_grad()
                with torch.no_grad():
                    _, h0 = initializer.forward(batch["d_init"])
                e_hat, _ = predictor.forward(batch["d"], h0)
                batch_loss = loss_function(e_hat, batch["e"])
                batch_phi = predictor.get_phi(t)
                (batch_loss + batch_phi).backward()
                opt_pred.step()
                predictor.set_lure_system()
                loss += batch_loss.item()
                phi += batch_phi.item()

            if epoch % PLOT_AFTER_EPOCHS == 0:
                fig = plot.plot_sequence(
                    [
                        e_hat[0, :].cpu().cpu().detach().numpy(),
                        batch["e"][0, :].cpu().detach().numpy(),
                    ],
                    0.01,
                    title="normalized output",
                    legend=[r"$\hat e$", r"$e$"],
                )
                tracker.track(ev.SaveFig("", fig, f"e_{epoch}"))

            if not predictor.check_constraints():
                problem_status = predictor.project_parameters()
                tracker.track(ev.Log("", f"Projecting parameters: {problem_status}"))
                predictor.set_lure_system()

            tracker.track(
                ev.Log(
                    "", f"{epoch}/{exp_config.epochs}\t l= {loss:.2f} \t phi= {phi:.2f}"
                )
            )
            tracker.track(
                ev.TrackMetrics(
                    "",
                    {
                        "epoch.loss.predictor": float(loss),
                        "epoch.phi.predictor": float(phi),
                    },
                    epoch,
                )
            )

            if sch_pred is not None:
                sch_pred.step(loss)

            if (epoch + 1) % increase_after_epochs == 0:
                t = t * increase_rate
                tracker.track(ev.Log("", f"Increase t by {increase_rate} to {t}"))

        return (initializer, predictor)
