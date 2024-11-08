from typing import Optional, Tuple

from ..configuration import (ModelConfiguration,
                             retrieve_model_class)
from .base import ConstrainedModule


def get_model_from_config(
    model_config: ModelConfiguration,
) -> Tuple[ConstrainedModule]:
    predictor: Optional[ConstrainedModule] = None
    initializer: Optional[ConstrainedModule] = None

    model_class = retrieve_model_class(model_config.m_class)

    init_params = model_config.parameters.model_copy()
    init_params.nd = init_params.nd + init_params.ne
    initializer = model_class(init_params)

    predictor = model_class(model_config.parameters)

    return (initializer, predictor)
