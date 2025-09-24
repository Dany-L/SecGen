from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from ..utils import transformation as trans
from . import base

class ReLiNetConfig(base.DynamicIdentificationConfig):
    num_layers: int = 1

class ReLiNet(base.DynamicIdentificationModel):
    CONFIG = ReLiNetConfig

    