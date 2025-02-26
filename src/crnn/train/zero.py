from typing import List, Optional, Tuple

import jax.numpy as jnp
import torch
from jax import grad
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from ..configuration.base import InputOutput
from ..configuration.experiment import BaseExperimentConfig
from ..models import base
from ..models.jax import base as base_jax
from ..models.torch import base as base_torch
from ..tracker import events as ev
from ..tracker.base import AggregatedTracker
from ..utils import plot
from ..utils import transformation as trans
from .base import PLOT_AFTER_EPOCHS, Armijo, InitPred


class ZeroInitPredictor(InitPred):

    def validate(
        self,
        predictor: base.DynamicIdentificationModel,
        val_loader: DataLoader,
        loss_function: nn.Module,
    ) -> Tuple[torch.Tensor, InputOutput]:
        with torch.no_grad():
            loss = torch.tensor(0.0)
            for batch in val_loader:
                e_hat, _ = predictor.forward(batch["d"])
                loss += loss_function(e_hat, batch["e"]).item()

            B, N_val, nd = batch["d"].shape
            _, _, ne = batch["e"].shape

        return (
            1 / len(val_loader) * loss,
            InputOutput(
                batch["d"].detach().numpy().reshape(N_val, nd),
                e_hat.detach().numpy().reshape(N_val, ne),
                batch["e"].detach().numpy().reshape(N_val, ne),
            ),
        )

    def train(
        self,
        models: List[base.DynamicIdentificationModel],
        loaders: List[DataLoader],
        exp_config: BaseExperimentConfig,
        optimizers: List[Optimizer],
        schedulers: List[lr_scheduler.ReduceLROnPlateau],
        loss_function: nn.Module,
        tracker: AggregatedTracker = AggregatedTracker(),
        initialize: bool = True,
    ) -> Tuple[Optional[base.DynamicIdentificationModel]]:
        train_loaders, validation_loaders = loaders
        _, train_loader = train_loaders
        _, validation_loader = validation_loaders
        _, predictor = models
        _, opt_pred = optimizers
        _, sch_pred = schedulers

        init_val_loss, io_sample = self.validate(
            predictor, validation_loader, loss_function
        )
        tracker.track(
            ev.SaveFig(
                "",
                plot.plot_sequence(
                    [io_sample.e, io_sample.e_hat],
                    0.1,
                    "validation sequence after initialization",
                    ["e", "e_hat"],
                ),
                "val_seq_init",
            )
        )
        tracker.track(
            ev.Log("", f"validation loss after initialization: {init_val_loss:.2f}")
        )
        tracker.track(ev.TrackMetrics("", {"init.val_loss.predictor": init_val_loss}))

        # predictor.train()
        t, increase_rate, increase_after_epochs = (
            exp_config.t,
            exp_config.increase_rate,
            exp_config.increase_after_epochs,
        )
        decrease_rate = 1 / 4
        alpha = 0.5
        dual_vars = [
            torch.tensor(1.0)
            for _ in range(
                len(predictor.sdp_constraints())
                + len(predictor.pointwise_constraints())
            )
        ]
        min_eigenvals = [
            torch.tensor(0.0)
            for _ in range(
                len(predictor.sdp_constraints())
                + len(predictor.pointwise_constraints())
            )
        ]

        steps_without_improvement, val_loss = 0, torch.tensor(0.0)
        for epoch in range(exp_config.epochs):
            loss, phi, projection_counter, num_backtracking_steps, grads = (
                0.0,
                0.0,
                0,
                0,
                torch.zeros((predictor.get_number_of_parameters(), 1)),
            )  # phi is barrier

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

                    elif isinstance(predictor, base_torch.ConstrainedModule):
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

                        # old_theta = theta.detach().clone()
                        batch_loss, batch_phi = f(theta), torch.tensor(0.0)

                        arm = Armijo(f, dF, 1)
                        # if step == 0 and epoch % PLOT_AFTER_EPOCHS == 0:
                        #     fig = arm.plot_step_size_function(theta)
                        #     tracker.track(ev.SaveFig('', fig, f'step_size_plot-{epoch}'))
                        dir = -dF(theta)
                        # print(dir)
                        s = arm.linesearch(theta, dir)
                        # new_theta = old_theta + s * dir
                        # opt_fcn.set_vec_pars_to_model(new_theta)

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
                        batch_phi = torch.tensor([0.0])
                        (batch_loss + batch_phi).backward()
                        opt_pred.step()

                elif exp_config.ensure_constrained_method == "project":
                    predictor.zero_grad()
                    e_hat, _ = predictor.forward(batch["d"])
                    batch_loss = loss_function(e_hat, batch["e"])
                    batch_phi = predictor.get_phi(t)
                    (batch_loss + batch_phi).backward()
                    opt_pred.step()

                    if not predictor.check_constraints():
                        problem_status = predictor.project_parameters()
                        # tracker.track(ev.Log("", f"{step}: Projecting parameters: {problem_status}"))
                        predictor.set_lure_system()
                        projection_counter += 1

                elif exp_config.ensure_constrained_method == "backtracking":
                    predictor.zero_grad()
                    e_hat, _ = predictor.forward(batch["d"])
                    batch_loss = loss_function(e_hat, batch["e"])
                    # batch_phi = torch.nn.functional.relu(predictor.get_phi(t))
                    batch_phi = predictor.get_phi(t)
                    (batch_loss + batch_phi).backward()
                    theta_old = trans.get_flat_parameters(predictor.parameters())

                    grads = torch.hstack(
                        [
                            p.grad.flatten().clone()
                            for p in predictor.parameters()
                            if p.grad is not None
                        ]
                    ).reshape(-1, 1)
                    # tracker.track(ev.Log("", f"||grad||: {torch.linalg.norm(grads):.2f} \t phi: {batch_phi.item():.2f}"))
                    opt_pred.step()

                    theta = trans.get_flat_parameters(predictor.parameters())
                    num_backtracking_steps = 0
                    while not predictor.check_constraints():
                        theta = alpha * theta + (1 - alpha) * theta_old.clone()
                        trans.set_vec_pars_to_model(predictor.parameters(), theta)
                        num_backtracking_steps += 1
                        if num_backtracking_steps > 100:
                            tracker.track(
                                ev.Log("", "Backtracking failed after 100 steps")
                            )
                            trans.set_vec_pars_to_model(
                                predictor.parameters(), theta_old
                            )
                            return (None, predictor)
                    # tracker.track(ev.Log("", f"Backtracking steps: {num_backtracking_steps}"))

                elif exp_config.ensure_constrained_method == "dual":

                    predictor.zero_grad()
                    e_hat, _ = predictor.forward(batch["d"])
                    batch_loss = loss_function(e_hat, batch["e"])
                    batch_phi = torch.tensor(0.0)
                    for idx, (lam_i, F_i) in enumerate(
                        zip(
                            dual_vars,
                            predictor.sdp_constraints()
                            + predictor.pointwise_constraints(),
                        )
                    ):
                        if len(F_i().shape) > 0:
                            min_eigenvals[idx] = -(
                                torch.min(torch.real(torch.linalg.eig(F_i())[0])) - 1e-3
                            )

                        else:
                            min_eigenvals[idx] = -F_i()
                        batch_phi += lam_i * min_eigenvals[idx]

                    (batch_loss + batch_phi).backward()

                    opt_pred.step()

                    with torch.no_grad():
                        for idx, (lam_i, min_eigenval) in enumerate(
                            zip(dual_vars, min_eigenvals)
                        ):
                            dual_vars[idx] = torch.max(
                                torch.tensor([lam_i - 0.01 * (-min_eigenval), 0])
                            )

                    grads = torch.hstack(
                        [
                            p.grad.flatten().clone()
                            for p in predictor.parameters()
                            if p.grad is not None
                        ]
                    ).reshape(-1, 1)

                else:
                    raise ValueError(
                        f"Unknown method to ensure constraints: {exp_config.ensure_constrained_method}"
                    )

                predictor.set_lure_system()
                loss += batch_loss.item()
                phi += batch_phi.item()

            if (
                not predictor.check_constraints()
                and not exp_config.ensure_constrained_method == "dual"
            ):
                problem_status = predictor.project_parameters()
                tracker.track(ev.Log("", f"Projecting parameters: {problem_status}"))
                predictor.set_lure_system()

            # evaluate parameters on validation data
            old_val_loss = val_loss.clone()
            val_loss, io_sample = self.validate(
                predictor, validation_loader, loss_function
            )
            if epoch % 10 == 0:
                tracker.track(
                    ev.SaveFig(
                        "",
                        plot.plot_sequence(
                            [io_sample.e, io_sample.e_hat],
                            0.1,
                            f"validation sequence e={epoch}",
                            ["e", "e_hat"],
                        ),
                        f"val_seq-{epoch}",
                    )
                )
            if old_val_loss <= val_loss:
                steps_without_improvement += 1

            tracker.track(
                ev.Log(
                    "",
                    f"{epoch}/{exp_config.epochs}\t l= {loss:.2f} \t val l= {val_loss:.2f} \t #projections= {projection_counter} \t backtracking steps= {num_backtracking_steps} \t ||grad||= {torch.linalg.norm(grads):.2f} \t phi= {phi:.2f} \t mean dual var={torch.mean(torch.tensor(dual_vars)):.2f} \t max ev: {torch.max(torch.tensor(min_eigenvals)):.2f}",
                )
            )
            tracker.track(
                ev.TrackMetrics(
                    "",
                    {
                        "epoch.loss.predictor": float(loss),
                        "epoch.phi.predictor": float(phi),
                        "epoch.val_loss.predictor": float(val_loss),
                    },
                    epoch,
                )
            )

            if steps_without_improvement > increase_after_epochs:
                # increase t
                t = t * increase_rate
                tracker.track(ev.Log("", f"Increase t by {increase_rate} to {t}"))
                # decrease lr
                for param_group in opt_pred.param_groups:
                    lr = param_group["lr"] * 1 / 4
                    param_group["lr"] = lr
                tracker.track(ev.Log("", f"Decrease lr by {decrease_rate} to {lr}"))
                steps_without_improvement = 0

        return (None, predictor)
