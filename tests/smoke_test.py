import sys
import os
import torch
import pytest
import json
import itertools
from pathlib import Path

from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
from crnn.train.base import train, continue_training
from crnn.evaluate import evaluate
from crnn.utils.base import get_device

torch.set_default_dtype(torch.double)


def get_test_parameters(configuration_file):
    """Load configuration and generate test parameters."""
    with open(configuration_file, mode="r") as f:
        config_dict = json.load(f)

    config = ExperimentConfig.from_template(ExperimentTemplate(**config_dict))
    model_names = config.m_names
    experiment_names = list(config.experiments.keys())

    # Determine available devices
    if get_device(gpu=True) == "cuda":
        devices = [True, False]
    else:
        devices = [False]

    # Create test parameters: (model_name, experiment_name, gpu)
    test_params = list(itertools.product(model_names, experiment_names, devices))
    test_ids = [f"{model}-{experiment}-{'gpu' if gpu else 'cpu'}" 
                for model, experiment, gpu in test_params]
    
    return test_params, test_ids


def pytest_generate_tests(metafunc):
    """Generate individual test functions for each model/experiment/device combination."""
    if "individual_test_params" in metafunc.fixturenames:
        # Load configuration file
        test_root_dir = Path(metafunc.config.rootpath) / "tests"
        configuration_file = test_root_dir / "config" / "config.json"
        
        try:
            test_params, test_ids = get_test_parameters(configuration_file)
            metafunc.parametrize("individual_test_params", test_params, ids=test_ids)
        except Exception as e:
            # If configuration loading fails, skip parametrization
            print(f"Warning: Could not load test parameters: {e}")
            metafunc.parametrize("individual_test_params", [("dummy", "dummy", False)], ids=["dummy"])


@pytest.mark.smoke
@pytest.mark.slow
def test_individual_model_training_and_evaluation(individual_test_params, configuration_file, 
                                                data_directory, temp_result_directory):
    """Individual test for each model/experiment/device combination."""
    model_name, experiment_name, gpu = individual_test_params
    
    if model_name == "dummy":
        pytest.skip("Configuration loading failed - skipping test")
    
    # Test training
    try:
        train(
            str(configuration_file),
            str(data_directory),
            temp_result_directory,
            model_name,
            experiment_name,
            gpu,
        )
    except Exception as e:
        pytest.fail(f"Training failed for {model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}: {str(e)}")
    
    # Test continue training
    try:
        continue_training(
            str(configuration_file),
            str(data_directory),
            temp_result_directory,
            model_name,
            experiment_name,
            gpu,
        )
    except Exception as e:
        pytest.fail(f"Continue training failed for {model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}: {str(e)}")
    
    # Test evaluation
    try:
        evaluate(
            str(configuration_file),
            str(data_directory),
            temp_result_directory,
            model_name,
            experiment_name,
        )
    except Exception as e:
        pytest.fail(f"Evaluation failed for {model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}: {str(e)}")


@pytest.mark.smoke
def test_configuration_loading(configuration_file):
    """Test that the configuration file can be loaded correctly."""
    assert configuration_file.exists(), f"Configuration file not found: {configuration_file}"
    
    with open(configuration_file, mode="r") as f:
        config_dict = json.load(f)
    
    config = ExperimentConfig.from_template(ExperimentTemplate(**config_dict))
    
    assert len(config.m_names) > 0, "No models found in configuration"
    assert len(config.experiments) > 0, "No experiments found in configuration"


@pytest.mark.smoke
def test_data_directory_exists(data_directory):
    """Test that the data directory exists."""
    assert data_directory.exists(), f"Data directory not found: {data_directory}"

