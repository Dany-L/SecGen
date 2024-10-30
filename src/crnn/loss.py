import torch.nn as nn


def get_loss_function(loss_function: str) -> nn.Module:
    if loss_function == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_function == "mse":
        return nn.MSELoss()
    elif loss_function == "nll":
        return nn.NLLLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
