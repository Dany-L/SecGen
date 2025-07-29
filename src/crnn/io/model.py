import os
from typing import List, Tuple

from ..configuration.experiment import BaseExperimentConfig, BaseModelConfig
from ..utils import base as utils
from ..models import base as base
from ..models.recurrent import BasicLstm


def get_model_from_config(
    model_config: BaseModelConfig,
) -> Tuple[base.DynamicIdentificationModel, base.DynamicIdentificationModel]:
    model_class = base.retrieve_model_class(model_config.m_class)

    if isinstance(model_class, BasicLstm):
        # to approximately match size of other models, two hidden states h and c
        model_config.parameters.nz = int(model_config.parameters.nz / 2)

    init_params = model_config.parameters.model_copy()
    init_params.nd = init_params.nd + init_params.ne
    initializer = model_class(init_params)

    predictor = model_class(model_config.parameters)

    return (initializer, predictor)


def set_parameters_to_train(
    model: base.DynamicIdentificationModel, parameter_names: List
) -> None:
    if isinstance(model, base.DynamicIdentificationModel):
        for n, p in model.named_parameters():
            if n in parameter_names:
                p.requires_grad = False
            else:
                p.requires_grad = True


def load_model(
    experiment_config: BaseExperimentConfig,
    initializer: base.DynamicIdentificationModel,
    predictor: base.DynamicIdentificationModel,
    model_name: str,
    result_directory: str,
) -> List[base.DynamicIdentificationModel]:
    if experiment_config.initial_hidden_state == "zero":
        initializer, predictor = None, base.load_model(
            predictor,
            os.path.join(
                result_directory,
                f'{utils.get_model_file_name("predictor", model_name)}.pth',
            ),
        )
    else:
        initializer, predictor = (
            base.load_model(
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


def copy_common_fields(
    source: BaseExperimentConfig, target: base.DynamicIdentificationConfig
) -> None:
    # Identify common fields
    common_fields = set(source.model_fields.keys()) & set(target.model_fields.keys())

    # Copy values from source to target
    for field in common_fields:
        setattr(target, field, getattr(source, field))
