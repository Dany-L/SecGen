from .models import base
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import nn
from .utils import plot
from .utils import base as utils
import matplotlib.pyplot as plt
from . import tracker as base_tracker
from typing import Optional
import time

def train_model(
        model: base.ConstrainedModule,
        train_loader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        loss_function: nn.Module,
        tracker: Optional[base_tracker.BaseTracker] = base_tracker.BaseTracker()
    ) -> base.ConstrainedModule:

    model.initialize_parameters()
    model.train()
    tracker.track(base_tracker.Start(''))

    for epoch in range(epochs):
        loss = 0
        for step, batch in enumerate(train_loader):
            model.zero_grad()
            model.set_lure_system()
            e_hat, _ = model.forward(batch['d'])
            batch_loss = loss_function(e_hat, batch['e'])
            batch_loss.backward()
            optimizer.step()
            model.check_constraints()
            loss+=batch_loss
        if epoch%100==0:
            fig = plot.plot_sequence([e_hat[0,:].detach().numpy(), batch['e'][0,:].detach().numpy()], 0.01, legend=['e_hat', 'e'])
            tracker.track(base_tracker.SaveFig('', fig, f'e_{epoch}'))
                        
        tracker.track(base_tracker.Log('', f'{epoch}/{epochs}\t l= {loss:.2f}\t feasible? {model.check_constraints()}'))
    tracker.track(base_tracker.Stop(''))
            
    return model