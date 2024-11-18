from typing import Optional, Tuple

from ..configuration.experiment import BaseModelConfig, retrieve_model_class
from .base import ConstrainedModule
from .recurrent import BasicLstm


def get_model_from_config(
    model_config: BaseModelConfig,
) -> Tuple[ConstrainedModule]:
    predictor: Optional[ConstrainedModule] = None
    initializer: Optional[ConstrainedModule] = None

    model_class = retrieve_model_class(model_config.m_class)

    if isinstance(model_class, BasicLstm):
        # to approximately match size of other models, two hidden states h and c
        model_config.parameters.nz = int(model_config.parameters.nz / 2)

    init_params = model_config.parameters.model_copy()
    init_params.nd = init_params.nd + init_params.ne
    initializer = model_class(init_params)

    predictor = model_class(model_config.parameters)

    return (initializer, predictor)
