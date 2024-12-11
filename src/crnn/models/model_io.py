import os
from typing import List, Optional, Tuple

from ..configuration.experiment import BaseExperimentConfig, BaseModelConfig
from ..utils import base as utils
from . import base as base
from .jax import base as base_jax
from .torch import base as base_torch
from .torch.recurrent import BasicLstm


def get_model_from_config(
    model_config: BaseModelConfig,
) -> Tuple[base.DynamicIdentificationModel]:
    predictor: Optional[base.DynamicIdentificationModel] = None
    initializer: Optional[base.DynamicIdentificationModel] = None

    # if "jax" in model_config.m_class:
    #     model_class = base.retrieve_model_class(model_config.m_class)
    # else:
    model_class = base.retrieve_model_class(model_config.m_class)

    if isinstance(model_class, BasicLstm):
        # to approximately match size of other models, two hidden states h and c
        model_config.parameters.nz = int(model_config.parameters.nz / 2)

    init_params = model_config.parameters.model_copy()
    init_params.nd = init_params.nd + init_params.ne
    initializer = model_class(init_params)

    predictor = model_class(model_config.parameters)

    return (initializer, predictor)


def load_model(
    experiment_config: BaseExperimentConfig,
    initializer: base.DynamicIdentificationModel,
    predictor: base.DynamicIdentificationModel,
    model_name: str,
    result_directory: str,
) -> List[base.DynamicIdentificationModel]:
    if isinstance(predictor, base_jax.ConstrainedModule):
        if experiment_config.initial_hidden_state == "zero":
            initializer, predictor = None, base_jax.load_model(
                predictor,
                os.path.join(
                    result_directory,
                    f'{utils.get_model_file_name("predictor", model_name)}.npz',
                ),
            )
        else:
            initializer, predictor = (
                base_jax.load_model(
                    model,
                    os.path.join(
                        result_directory,
                        f"{utils.get_model_file_name(name, model_name)}.npz",
                    ),
                )
                for model, name in zip(
                    [initializer, predictor], ["initializer", "predictor"]
                )
            )

        return [initializer, predictor]
    else:
        if experiment_config.initial_hidden_state == "zero":
            initializer, predictor = None, base_torch.load_model(
                predictor,
                os.path.join(
                    result_directory,
                    f'{utils.get_model_file_name("predictor", model_name)}.pth',
                ),
            )
        else:
            initializer, predictor = (
                base_torch.load_model(
                    model,
                    os.path.join(
                        result_directory,
                        f"{utils.get_model_file_name(name, model_name)}.pth",
                    ),
                )
                for model, name in zip(
                    [initializer, predictor], ["initializer", "predictor"]
                )
            )

        return [initializer, predictor]
