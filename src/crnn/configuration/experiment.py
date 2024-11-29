import itertools
import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from ..additional_tests import AdditionalTestConfig, retrieve_additional_test_class
from ..metrics import MetricConfig, retrieve_metric_class
from ..models.base import DynamicIdentificationConfig, retrieve_model_class
from ..tracker.base import BaseTrackerConfig, retrieve_tracker_class


class ExperimentAdditionalTest(BaseModel):
    test_class: str
    parameters: Dict[str, Any]


class ExperimentMetric(BaseModel):
    metric_class: str
    parameters: Dict[str, Any]


class ExperimentTracker(BaseModel):
    tracker_class: str
    parameters: Dict[str, Any]


class ExperimentModel(BaseModel):
    m_class: str
    m_short_name: str
    parameters: Dict[str, Any]


class BaseAdditionalTestConfig(BaseModel):
    test_class: str
    parameters: AdditionalTestConfig


class BaseTrackerConfig(BaseModel):
    tracker_class: str
    parameters: BaseTrackerConfig


class BaseMetricConfig(BaseModel):
    metric_class: str
    parameters: MetricConfig


class BaseModelConfig(BaseModel):
    m_class: str
    parameters: DynamicIdentificationConfig


class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd"]
    learning_rate: float


class HorizonsConfig(BaseModel):
    training: int
    testing: int


class ExperimentSettings(BaseModel):
    experiment_base_name: str
    metrics: Dict[str, ExperimentMetric]
    trackers: Dict[str, ExperimentTracker]
    additional_tests: Dict[str, ExperimentAdditionalTest]
    static_parameters: Dict[str, Any]
    flexible_parameters: Dict[str, List[Any]]


class ExperimentTemplate(BaseModel):
    settings: ExperimentSettings
    models: List[ExperimentModel]


class BaseExperimentConfig(BaseModel):
    epochs: int
    eps: float
    optimizer: OptimizerConfig
    dt: float
    nz: int
    batch_size: int
    window: int
    loss_function: Literal["mse"]
    input_names: List[str]
    output_names: List[str]
    horizons: HorizonsConfig
    initial_hidden_state: Literal["zero", "joint", "separate"]
    t: float = 1.0
    increase_rate: float = 10.0
    increase_after_epochs: int = 100
    debug: bool = False
    ensure_constrained_method: Literal["armijo"] = None


class ExperimentConfig(BaseModel):
    experiments: Dict[str, BaseExperimentConfig]
    models: Dict[str, BaseModelConfig]
    trackers: Dict[str, BaseTrackerConfig]
    metrics: Dict[str, BaseMetricConfig]
    additional_tests: Dict[str, BaseAdditionalTestConfig]
    m_names: List[str]

    @classmethod
    def from_template(cls, template: ExperimentTemplate) -> "ExperimentConfig":
        experiments: Dict[str, BaseExperimentConfig] = dict()
        models: Dict[str, BaseModelConfig] = dict()

        nd, ne = len(template.settings.static_parameters["input_names"]), len(
            template.settings.static_parameters["output_names"]
        )
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
                    experiment_base_name += "-" + "_".join(map(str, param_value))
                else:
                    if isinstance(param_value, float):
                        experiment_base_name += f"-{param_value:g.2}".replace(".", "")
                    else:
                        experiment_base_name += f"-{param_value}".replace(".", "")

            experiment_config = BaseExperimentConfig(
                **{**static_experiment_params, **flexible_experiment_params}
            )
            experiments[experiment_base_name] = experiment_config

            for model in template.models:
                model_class = retrieve_model_class(model.m_class)
                models[f"{experiment_base_name}-{model.m_short_name}"] = (
                    BaseModelConfig(
                        m_class=model.m_class,
                        parameters=model_class.CONFIG(
                            **{
                                **model.parameters,
                                **static_experiment_params,
                                **flexible_experiment_params,
                                "nd": nd,
                                "ne": ne,
                            }
                        ),
                    )
                )

        model_names = []
        for model in template.models:
            model_names.append(model.m_short_name)

        metrics = dict()
        for name, metric in template.settings.metrics.items():
            metric_class = retrieve_metric_class(metric.metric_class)
            metrics[name] = BaseMetricConfig(
                metric_class=metric.metric_class,
                parameters=metric_class.CONFIG(**metric.parameters),
            )

        additional_tests = dict()
        for name, additional_test in template.settings.additional_tests.items():
            additional_test_class = retrieve_additional_test_class(
                additional_test.test_class
            )
            additional_tests[name] = BaseAdditionalTestConfig(
                test_class=additional_test.test_class,
                parameters=additional_test_class.CONFIG(**additional_test.parameters),
            )
        trackers = dict()
        for name, tracker in template.settings.trackers.items():
            tracker_class = retrieve_tracker_class(tracker.tracker_class)
            trackers[name] = BaseTrackerConfig(
                tracker_class=tracker.tracker_class,
                parameters=tracker_class.CONFIG(**tracker.parameters),
            )

        return cls(
            experiments=experiments,
            models=models,
            metrics=metrics,
            m_names=model_names,
            additional_tests=additional_tests,
            trackers=trackers,
        )


def load_configuration(config_file_name: str) -> ExperimentConfig:

    with open(config_file_name, "r") as file:
        config_data = json.load(file)
    return ExperimentConfig.from_template(ExperimentTemplate(**config_data))
