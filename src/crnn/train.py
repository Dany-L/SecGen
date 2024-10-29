from .models import base
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def train_model(
        model: base.ConstrainedModule,
        train_loader: DataLoader,
        epochs: int,
        optimizer: Optimizer
)