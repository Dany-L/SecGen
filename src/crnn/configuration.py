import json
import os
from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Tuple, Type
import itertools

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from .models.base import DynamicIdentificationConfig, DynamicIdentificationModel
from .metrics import MetricConfig, retrieve_metric_class
import argparse

CONFIG_FILE_ENV_VAR = "CONFIGURATION"
DATASET_DIR_ENV_VAR = "DATASET_DIRECTORY"
RESULT_DIR_ENV_VAR = "RESULT_DIRECTORY"
FIG_FOLDER_NAME = "fig"





@dataclass
class NormalizationParameters:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


@dataclass
class Normalization:
    input: NormalizationParameters
    output: NormalizationParameters

class Duration(BaseModel):
    duration_str: str
    duration: float

class ExperimentMetric(BaseModel):
    metric_class: str
    parameters: Dict[str, Any]
class BaseMetricConfig(BaseModel):
    metric_class: str
    parameters: MetricConfig

class ResultConfig(BaseModel):
    init_training_time: Duration
    pred_training_time: Duration
    number_of_parameters: int
    metrics: Dict[str, float]


class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd"]
    learning_rate: float


class HorizonsConfig(BaseModel):
    training: int
    testing: int


class ModelTemplate(BaseModel):
    m_class: str
    m_short_name:str
    parameters: Dict[str, Any]

class ModelConfiguration(BaseModel):
    m_class: str
    parameters: DynamicIdentificationConfig

class ExperimentSettings(BaseModel):
    experiment_base_name: str
    metrics: Dict[str, ExperimentMetric]
    static_parameters: Dict[str, Any]
    flexible_parameters: Dict[str, List[Any]]
class ExperimentTemplate(BaseModel):
    settings: ExperimentSettings
    models: List[ModelTemplate]

class ExperimentBaseConfig(BaseModel):
    epochs: int
    eps: float
    optimizer: OptimizerConfig
    num_layer: int
    dt: float
    nz: int
    batch_size: int
    window: int
    loss_function: Literal["mse"]
    input_names: List[str]
    output_names: List[str]
    horizons: HorizonsConfig
    initial_hidden_state: Literal["zero", "joint", "separate"]
class ExperimentConfig(BaseModel):
    experiments: Dict[str, ExperimentBaseConfig]
    models: Dict[str, ModelConfiguration]
    metrics: Dict[str, BaseMetricConfig]

    @classmethod
    def from_template(cls, template: ExperimentTemplate) -> 'ExperimentConfig':
        experiments: Dict[str, ExperimentBaseConfig] = dict()

        static_experiment_params = template.settings.static_parameters
        
        for combination in itertools.product(
            *list(template.settings.flexible_parameters.values())
        ):
            experiment_base_name = template.settings.experiment_base_name
            flexible_experiment_params = dict()
            for param_name, param_value in zip(
                template.settings.flexible_parameters.keys(), combination
            ):
                flexible_experiment_params[param_name] = param_value
                if issubclass(type(param_value), list):
                    experiment_base_name += '-' + '_'.join(map(str, param_value))
                else:
                    if isinstance(param_value, float):
                        experiment_base_name += f'-{param_value:g.2}'.replace('.', '')
                    else:
                        experiment_base_name += f'-{param_value}'.replace('.', '')

            experiment_config = ExperimentBaseConfig(**{**static_experiment_params, **flexible_experiment_params})
            experiments[experiment_base_name] = experiment_config

        nd, ne = len(experiment_config.input_names), len(experiment_config.output_names)
        models: Dict[str,ModelConfiguration] = dict()
        for model in template.models:
            model_class = retrieve_model_class(model.m_class)  
            models[model.m_short_name] = ModelConfiguration(
                m_class=model.m_class,
                parameters=model_class.CONFIG(**{**model.parameters, **static_experiment_params, **flexible_experiment_params, 'nd': nd, 'ne': ne})
            )
        
        metrics = dict()
        for name, metric in template.settings.metrics.items():
            metric_class = retrieve_metric_class(metric.metric_class)
            metrics[name] = BaseMetricConfig(
                metric_class=metric.metric_class,
                parameters=metric_class.CONFIG(
                    **metric.parameters
                ),
            )
            
        return cls(experiments=experiments, models=models, metrics=metrics)

        


def load_configuration(config_file_name: str) -> ExperimentConfig:
    
    with open(config_file_name, "r") as file:
        config_data = json.load(file)
    return ExperimentConfig.from_template(ExperimentTemplate(**config_data))


def retrieve_model_class(model_class_string: str) -> Type[DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, DynamicIdentificationModel):
        raise ValueError(f'{cls} is not a subclass of DynamicIdentificationModel')
    return cls  # type: ignore

def parse_input(parser:argparse.ArgumentParser)-> Tuple[str, str]:
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("model", type=str, help="Name of the model to train")
    args = parser.parse_args()
    return (args.model, args.experiment_name)