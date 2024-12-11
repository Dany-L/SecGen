from typing import List, Optional, Tuple

import jax.numpy as jnp
import torch
from jax import grad
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from ..configuration.experiment import BaseExperimentConfig
from ..models import base
from ..models.jax import base as base_jax
from ..models.torch import base as base_torch
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker
from ..utils import plot
from .base import PLOT_AFTER_EPOCHS, Armijo, InitPred


class ZeroInitPredictor(InitPred):

    def train(
        models: Tuple[base.DynamicIdentificationModel],
        loaders: Tuple[DataLoader],
        exp_config: BaseExperimentConfig,
        optimizers: List[Optimizer],
        schedulers: List[lr_scheduler.ReduceLROnPlateau],
        loss_function: nn.Module,
        tracker: AggregatedTracker = AggregatedTracker(),
    ) -> Tuple[Optional[base.DynamicIdentificationModel]]:
        _, train_loader = loaders
        _, predictor = models
        _, opt_pred = optimizers
        _, sch_pred = schedulers
        predictor.initialize_parameters()
        predictor.set_lure_system()
        # predictor.train()
        t, increase_rate, increase_after_epochs = (
            exp_config.t,
            exp_config.increase_rate,
            exp_config.increase_after_epochs,
        )

        for epoch in range(exp_config.epochs):
            loss, phi = 0.0, 0.0  # phi is barrier

            for step, batch in enumerate(train_loader):
                # predictor.zero_grad()
                if exp_config.ensure_constrained_method == "armijo":
                    if isinstance(predictor, base_jax.ConstrainedModule):
                        d, e, x0 = (
                            batch["d"].cpu().detach().numpy(),
                            batch["e"].cpu().detach().numpy(),
                            None,
                        )
                        theta = predictor.theta

                        def f(theta):
                            e_hat, _ = predictor.forward(d, x0, theta)
                            loss = jnp.mean((e_hat - e) ** 2)
                            phi = predictor.get_phi(t, theta)
                            return loss + phi

                        dF = grad(f)
                        armijo = Armijo(f, dF)
                        s = armijo.linesearch(theta, -dF(theta))
                        # print(f'step size: {s}')
                        # print(f'gradient size: {jnp.linalg.norm(dF(theta))}')
                        if step == 0 and epoch % PLOT_AFTER_EPOCHS == 0:
                            fig = armijo.plot_step_size_function(theta)
                            tracker.track(
                                ev.SaveFig("", fig, f"step_size_plot-{epoch}")
                            )

                        theta = base_jax.update(theta, dF, s)
                        predictor.theta = theta
                        # print(f'constraints satisfied? {predictor.check_constraints(theta)}')

                        batch_loss = torch.from_dlpack(f(theta))
                        batch_phi = torch.tensor([0.0])

                        e_hat, _ = predictor.forward(d, x0, theta)

                        if step % 100 == 0:
                            fig = plot.plot_sequence(
                                [
                                    e_hat[0, :],
                                    e[0, :],
                                ],
                                0.01,
                                title="normalized output",
                                legend=[r"$\hat e$", r"$e$"],
                            )
                            tracker.track(ev.SaveFig("", fig, f"e_{epoch}-step_{step}"))

                    elif isinstance(predictor, base_torch.DynamicIdentificationModel):
                        opt_fcn = base_torch.OptFcn(
                            batch["d"], batch["e"], predictor, t, loss=loss_function
                        )

                        def f(theta: torch.Tensor) -> torch.Tensor:
                            return opt_fcn.f(theta)

                        def dF(theta: torch.Tensor) -> torch.Tensor:
                            return opt_fcn.dF(theta).reshape((-1, 1))

                        theta = torch.hstack(
                            [
                                p.flatten().clone()
                                for p in predictor.parameters()
                                if p is not None
                            ]
                        ).reshape(-1, 1)

                        old_theta = theta.clone()
                        batch_loss, batch_phi = f(theta), torch.tensor(0.0)

                        arm = Armijo(f, dF, 1)
                        # if step == 0 and epoch % PLOT_AFTER_EPOCHS == 0:
                        #     fig = arm.plot_step_size_function(old_theta)
                        #     tracker.track(ev.SaveFig('', fig, f'step_size_plot-{epoch}'))
                        dir = -dF(old_theta)
                        s = arm.linesearch(old_theta, dir)
                        new_theta = old_theta + s * dir
                        opt_fcn.set_vec_pars_to_model(new_theta)

                        tracker.track(
                            ev.TrackMetrics(
                                "",
                                {
                                    "loss.step": float(batch_loss),
                                    "stepsize.step": float(s),
                                },
                                epoch * len(train_loader) + step,
                            )
                        )

                else:
                    e_hat, _ = predictor.forward(batch["d"])
                    batch_loss = loss_function(e_hat, batch["e"])
                    batch_phi = predictor.get_phi(t)
                    (batch_loss + batch_phi).backward()
                    opt_pred.step()

                predictor.set_lure_system()
                loss += batch_loss.item()
                phi += batch_phi.item()

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

        return (None, predictor)
