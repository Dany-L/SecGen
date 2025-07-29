import sys
import os
import torch
import pytest
from pathlib import Path

from utils import run_tests
from crnn.configuration.base import IN_DISTRIBUTION_FOLDER_NAME, PROCESSED_FOLDER_NAME
from crnn.configuration.experiment import ExperimentConfig, ExperimentTemplate
from crnn.train.base import train, continue_training
from crnn.evaluate import evaluate
from crnn.utils.base import get_device

torch.set_default_dtype(torch.double)


# List of nonlinear benchmark datasets
DATASET_NAMES = [
    'CED', 
    'Cascaded_Tanks', 
    'EMPS', 
    'Silverbox', 
    'WienerHammerBenchMark', 
    'ParWH', 
    'F16'
]


@pytest.fixture(scope="session")
def benchmark_root_dir():
    """Root directory for benchmark tests."""
    return Path(__file__).parent


@pytest.fixture(scope="function")
def benchmark_result_directory(benchmark_root_dir, tmp_path):
    """Temporary result directory for benchmark tests."""
    return str(tmp_path)


def get_benchmark_paths(dataset_name: str, benchmark_root_dir: Path):
    """Get configuration and data paths for a benchmark dataset."""
    configuration_file = benchmark_root_dir / "config" / "nonlinear_benchmarks" / f"{dataset_name}.json"
    data_directory = benchmark_root_dir / "_tmp" / "data" / dataset_name / IN_DISTRIBUTION_FOLDER_NAME / PROCESSED_FOLDER_NAME
    return configuration_file, data_directory


def get_benchmark_test_parameters(configuration_file: Path):
    """Load configuration and generate test parameters for benchmark dataset."""
    if not configuration_file.exists():
        return [], []
    
    try:
        with open(configuration_file, mode="r") as f:
            import json
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
        import itertools
        test_params = list(itertools.product(model_names, experiment_names, devices))
        test_ids = [f"{model}-{experiment}-{'gpu' if gpu else 'cpu'}" 
                    for model, experiment, gpu in test_params]
        
        return test_params, test_ids
    except Exception:
        return [], []


def pytest_generate_tests(metafunc):
    """Generate individual test functions for each dataset and model combination."""
    if "benchmark_dataset_params" in metafunc.fixturenames:
        # Generate tests for each dataset
        metafunc.parametrize("benchmark_dataset_params", DATASET_NAMES, ids=DATASET_NAMES)
    
    elif "individual_benchmark_params" in metafunc.fixturenames:
        # Generate individual tests for each model/experiment/device combination across all datasets
        all_params = []
        all_ids = []
        
        benchmark_root_dir = Path(metafunc.config.rootpath) / "tests"
        
        for dataset_name in DATASET_NAMES:
            configuration_file, _ = get_benchmark_paths(dataset_name, benchmark_root_dir)
            test_params, test_ids = get_benchmark_test_parameters(configuration_file)
            
            # Add dataset name to parameters and IDs
            for params, test_id in zip(test_params, test_ids):
                all_params.append((dataset_name, *params))
                all_ids.append(f"{dataset_name}-{test_id}")
        
        if all_params:
            metafunc.parametrize("individual_benchmark_params", all_params, ids=all_ids)
        else:
            # Fallback if no configurations found
            metafunc.parametrize("individual_benchmark_params", [("dummy", "dummy", "dummy", False)], ids=["dummy"])


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_dataset_complete(benchmark_dataset_params, benchmark_root_dir, benchmark_result_directory):
    """Test complete benchmark dataset using the legacy run_tests function."""
    dataset_name = benchmark_dataset_params
    
    print(f'\n--- Dataset {dataset_name} ---')
    
    configuration_file, data_directory = get_benchmark_paths(dataset_name, benchmark_root_dir)
    
    # Skip if configuration file doesn't exist
    if not configuration_file.exists():
        pytest.skip(f"Configuration file not found: {configuration_file}")
    
    # Create data directory
    data_directory.mkdir(parents=True, exist_ok=True)
    
    try:
        run_tests(str(configuration_file), str(data_directory), benchmark_result_directory)
        print(f"✓ Completed benchmark dataset: {dataset_name}")
    except Exception as e:
        pytest.fail(f"Benchmark dataset {dataset_name} failed: {str(e)}")


@pytest.mark.benchmark
@pytest.mark.slow
def test_individual_benchmark_training_and_evaluation(individual_benchmark_params, benchmark_root_dir, benchmark_result_directory):
    """Individual test for each dataset/model/experiment/device combination."""
    dataset_name, model_name, experiment_name, gpu = individual_benchmark_params
    
    if dataset_name == "dummy":
        pytest.skip("No benchmark configurations found - skipping test")
    
    print(f"\nTesting {dataset_name}/{model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}")
    
    configuration_file, data_directory = get_benchmark_paths(dataset_name, benchmark_root_dir)
    
    # Skip if configuration file doesn't exist
    if not configuration_file.exists():
        pytest.skip(f"Configuration file not found: {configuration_file}")
    
    # Create data directory
    data_directory.mkdir(parents=True, exist_ok=True)
    
    # Test training
    try:
        train(
            str(configuration_file),
            str(data_directory),
            benchmark_result_directory,
            model_name,
            experiment_name,
            gpu,
        )
        print(f"✓ Training completed for {dataset_name}/{model_name}/{experiment_name}")
    except Exception as e:
        pytest.fail(f"Training failed for {dataset_name}/{model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}: {str(e)}")
    
    # Test continue training
    try:
        continue_training(
            str(configuration_file),
            str(data_directory),
            benchmark_result_directory,
            model_name,
            experiment_name,
            gpu,
        )
        print(f"✓ Continue training completed for {dataset_name}/{model_name}/{experiment_name}")
    except Exception as e:
        pytest.fail(f"Continue training failed for {dataset_name}/{model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}: {str(e)}")
    
    # Test evaluation
    try:
        evaluate(
            str(configuration_file),
            str(data_directory),
            benchmark_result_directory,
            model_name,
            experiment_name,
        )
        print(f"✓ Evaluation completed for {dataset_name}/{model_name}/{experiment_name}")
    except Exception as e:
        pytest.fail(f"Evaluation failed for {dataset_name}/{model_name}/{experiment_name}/{'gpu' if gpu else 'cpu'}: {str(e)}")


@pytest.mark.benchmark
def test_benchmark_configurations_exist(benchmark_root_dir):
    """Test that benchmark configuration files exist."""
    config_dir = benchmark_root_dir / "config" / "nonlinear_benchmarks"
    
    if not config_dir.exists():
        pytest.skip(f"Benchmark config directory not found: {config_dir}")
    
    existing_configs = []
    missing_configs = []
    
    for dataset_name in DATASET_NAMES:
        config_file = config_dir / f"{dataset_name}.json"
        if config_file.exists():
            existing_configs.append(dataset_name)
        else:
            missing_configs.append(dataset_name)
    
    print(f"\n📁 Existing benchmark configs: {existing_configs}")
    if missing_configs:
        print(f"⚠️  Missing benchmark configs: {missing_configs}")
    
    assert len(existing_configs) > 0, "No benchmark configuration files found"