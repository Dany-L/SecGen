"""
Tests for the preprocessing functionality.

This module tests all aspects of the preprocessing pipeline including:
- System identification and transient time calculation
- Horizon and window calculation  
- Normalization parameter calculation
- Data splitting (placeholder functionality)
- Integration with the configuration system
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np

from crnn.preprocessing.base import (
    preprocess,
    check_preprocessing_completed,
    calculate_system_identification,
    calculate_horizon_and_window,
    calculate_normalization_parameters,
    perform_data_splitting
)
from crnn.configuration.base import (
    SystemIdentificationResults,
    PreprocessingResults,
    Normalization,
    NormalizationParameters
)

import pytest
import numpy as np
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

# Add scripts directory to path for importing preprocessing
test_dir = Path(__file__).parent
scripts_dir = test_dir.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from crnn.preprocessing.base import (
    preprocess,
    check_preprocessing_completed,
    calculate_system_identification,
    calculate_horizon_and_window,
    calculate_normalization_parameters,
    perform_data_splitting
)

from crnn.configuration.base import (
    SystemIdentificationResults,
    PreprocessingResults,
    Normalization,
    NormalizationParameters
)
from crnn.configuration.experiment import load_configuration
from crnn.tracker.base import AggregatedTracker
from crnn.io.data import get_result_directory_name


class TestSystemIdentification:
    """Test system identification functionality."""
    
    def test_calculate_system_identification(self, data_directory):
        """Test N4SID system identification calculation."""
        # Create mock training data with some structure
        n_samples = 500
        n_inputs = 1
        n_outputs = 1
        
        # Create data with some dynamics to ensure non-zero transient time
        t = np.linspace(0, 10, n_samples)
        u = np.sin(0.5 * t).reshape(-1, 1)
        y = 0.8 * u + 0.1 * np.random.randn(n_samples, 1)
        
        train_inputs = [u]
        train_outputs = [y]
        
        dt = 0.01
        nz = 4
        tracker = AggregatedTracker()
        
        # Test system identification
        result = calculate_system_identification(
            train_inputs, train_outputs, dt, nz, tracker
        )
        
        # Verify result type and structure
        assert isinstance(result, SystemIdentificationResults)
        assert result.ss is not None
        assert isinstance(result.transient_time, int)
        assert result.transient_time >= 0  # Allow 0 for very simple systems
        assert isinstance(result.fit, float) or result.fit is None
        assert isinstance(result.rmse, np.ndarray) or result.rmse is None

    def test_calculate_system_identification_with_real_data(self, data_directory):
        """Test system identification with actual test data."""
        # Load real test data if available
        train_dir = data_directory / "train"
        if not train_dir.exists():
            pytest.skip("No test training data available")
        
        # Load available CSV files
        csv_files = list(train_dir.glob("*.csv"))
        if not csv_files:
            pytest.skip("No CSV files in test data directory")
        
        # Read the first available file using basic CSV reading
        import csv
        with open(csv_files[0], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            data = []
            for row in reader:
                data.append([float(x) for x in row])
        
        data = np.array(data)
        
        # Assume first column is input, second is output
        if data.shape[1] < 2:
            pytest.skip("Test data doesn't have enough columns")
        
        train_inputs = [data[:, 0].reshape(-1, 1)]
        train_outputs = [data[:, 1].reshape(-1, 1)]
        
        dt = 0.01
        nz = 2  # Use smaller state space for test data
        tracker = AggregatedTracker()
        
        result = calculate_system_identification(
            train_inputs, train_outputs, dt, nz, tracker
        )
        
        assert isinstance(result, SystemIdentificationResults)
        assert result.transient_time >= 0


class TestHorizonWindowCalculation:
    """Test horizon and window calculation."""
    
    def test_calculate_horizon_and_window_default(self):
        """Test default horizon and window calculation."""
        transient_time = 100
        horizon, window = calculate_horizon_and_window(transient_time)
        
        assert horizon == transient_time
        assert window == int(0.1 * horizon)
        assert window == 10
    
    def test_calculate_horizon_and_window_custom_ratio(self):
        """Test horizon and window calculation with custom ratio."""
        transient_time = 200
        window_ratio = 0.2
        horizon, window = calculate_horizon_and_window(transient_time, window_ratio)
        
        assert horizon == transient_time
        assert window == int(window_ratio * horizon)
        assert window == 40
    
    def test_calculate_horizon_and_window_edge_cases(self):
        """Test edge cases for horizon and window calculation."""
        # Very small transient time
        horizon, window = calculate_horizon_and_window(1)
        assert horizon == 1
        assert window == 0
        
        # Large transient time
        horizon, window = calculate_horizon_and_window(10000)
        assert horizon == 10000
        assert window == 1000


class TestNormalizationParameters:
    """Test normalization parameter calculation."""
    
    def test_calculate_normalization_parameters(self):
        """Test normalization parameter calculation."""
        # Create test data with known statistics
        np.random.seed(42)
        
        # Create input data with known mean and std
        input_data = np.random.normal(5.0, 2.0, (100, 2))
        output_data = np.random.normal(-1.0, 0.5, (100, 1))
        
        train_inputs = [input_data]
        train_outputs = [output_data]
        tracker = AggregatedTracker()
        
        result = calculate_normalization_parameters(
            train_inputs, train_outputs, tracker
        )
        
        # Verify result type
        assert isinstance(result, Normalization)
        assert isinstance(result.input, NormalizationParameters)
        assert isinstance(result.output, NormalizationParameters)
        
        # Check dimensions
        assert result.input.mean.shape == (2,)
        assert result.input.std.shape == (2,)
        assert result.output.mean.shape == (1,)
        assert result.output.std.shape == (1,)
        
        # Check that means are approximately correct
        np.testing.assert_allclose(result.input.mean, input_data.mean(axis=0), rtol=0.1)
        np.testing.assert_allclose(result.output.mean, output_data.mean(axis=0), rtol=0.1)
    
    def test_normalization_validation(self):
        """Test that normalization validation works correctly."""
        # Create data that should normalize properly
        train_inputs = [np.random.randn(50, 1)]
        train_outputs = [np.random.randn(50, 1)]
        tracker = AggregatedTracker()
        
        # Should not raise an assertion error
        result = calculate_normalization_parameters(
            train_inputs, train_outputs, tracker
        )
        
        assert isinstance(result, Normalization)


class TestDataSplitting:
    """Test data splitting functionality."""
    
    def test_perform_data_splitting_placeholder(self):
        """Test the data splitting placeholder functionality."""
        # Create mock data
        train_inputs = [np.random.randn(100, 2)]
        train_outputs = [np.random.randn(100, 1)]
        val_inputs = [np.random.randn(50, 2)]
        val_outputs = [np.random.randn(50, 1)]
        
        window = 10
        horizon = 100
        
        result = perform_data_splitting(
            train_inputs, train_outputs, val_inputs, val_outputs,
            window, horizon
        )
        
        # Check that the placeholder returns the original data
        assert result["train_inputs"] == train_inputs
        assert result["train_outputs"] == train_outputs
        assert result["val_inputs"] == val_inputs
        assert result["val_outputs"] == val_outputs
        assert result["ood_inputs"] is None
        assert result["ood_outputs"] is None


class TestPreprocessingCompletion:
    """Test preprocessing completion checking."""
    
    def test_check_preprocessing_completed_missing_files(self, temp_result_directory):
        """Test checking preprocessing completion with missing files."""
        # Empty directory should return False
        result = check_preprocessing_completed(temp_result_directory)
        assert result is False
    
    def test_check_preprocessing_completed_with_files(self, temp_result_directory):
        """Test checking preprocessing completion with required files."""
        # Create mock files
        init_file = os.path.join(temp_result_directory, "initialization.pkl")
        norm_file = os.path.join(temp_result_directory, "normalization.pkl")
        
        # Create empty files
        Path(init_file).touch()
        Path(norm_file).touch()
        
        result = check_preprocessing_completed(temp_result_directory)
        assert result is True


class TestCompletePreprocessingWorkflow:
    """Test the complete preprocessing workflow."""
    
    @pytest.fixture
    def mock_config_file(self, temp_result_directory):
        """Create a temporary configuration file for testing."""
        config_content = {
            "experiments": {
                "test_experiment": {
                    "dt": 0.01,
                    "nz": 4,
                    "input_names": ["u_1"],
                    "output_names": ["y_1"]
                }
            },
            "models": {
                "test_experiment-test_model": {
                    "m_class": "test_class"
                }
            },
            "trackers": {}
        }
        
        import json
        config_file = os.path.join(temp_result_directory, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_content, f)
        
        return config_file
    
    @pytest.fixture
    def mock_dataset_dir(self, temp_result_directory):
        """Create a temporary dataset directory with test data."""
        dataset_dir = os.path.join(temp_result_directory, "dataset")
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "validation")
        
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        
        # Create mock CSV files
        import csv
        
        # Training data
        train_file = os.path.join(train_dir, "train_data.csv")
        with open(train_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['u_1', 'y_1'])  # Header
            train_data = np.column_stack([np.random.randn(100), np.random.randn(100)])
            for row in train_data:
                writer.writerow(row)
        
        # Validation data
        val_file = os.path.join(val_dir, "val_data.csv")
        with open(val_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['u_1', 'y_1'])  # Header
            val_data = np.column_stack([np.random.randn(50), np.random.randn(50)])
            for row in val_data:
                writer.writerow(row)
        
        return dataset_dir
    
    @patch('crnn.preprocessing.base.load_configuration')
    @patch('crnn.preprocessing.base.get_trackers_from_config')
    @patch('crnn.preprocessing.base.load_data')
    @patch('crnn.preprocessing.base.load_initialization')
    def test_run_preprocessing_mock(
        self, 
        mock_load_init,
        mock_load_data,
        mock_get_trackers,
        mock_load_config,
        temp_result_directory
    ):
        """Test the complete preprocessing workflow with mocked dependencies."""
        
        # Setup mocks
        mock_config = MagicMock()
        mock_config.experiments = {
            "test_exp": MagicMock(
                dt=0.01, 
                nz=4, 
                input_names=["u_1"], 
                output_names=["y_1"]
            )
        }
        mock_config.trackers = {}
        mock_load_config.return_value = mock_config
        
        mock_get_trackers.return_value = []
        
        # Mock training data
        train_inputs = [np.random.randn(100, 1)]
        train_outputs = [np.random.randn(100, 1)]
        val_inputs = [np.random.randn(50, 1)]
        val_outputs = [np.random.randn(50, 1)]
        
        mock_load_data.side_effect = [
            (train_inputs, train_outputs),
            (val_inputs, val_outputs)
        ]
        
        # Mock initialization data (empty initially)
        mock_init_data = MagicMock()
        mock_init_data.data = {}
        mock_load_init.return_value = mock_init_data
        
        # Mock the N4SID calculation
        with patch('crnn.preprocessing.base.base.run_n4sid') as mock_n4sid, \
             patch('crnn.preprocessing.base.get_transient_time') as mock_transient:
            
            mock_n4sid.return_value = (MagicMock(), 0.85, 0.15)
            mock_transient.return_value = np.array([50.0])
            
            # Run preprocessing
            result = preprocess(
                "dummy_config.json",
                "dummy_dataset",
                temp_result_directory,
                "test_model",
                "test_exp"
            )
            
            # Verify result structure
            assert isinstance(result, PreprocessingResults)
            assert isinstance(result.system_identification, SystemIdentificationResults)
            assert isinstance(result.normalization, Normalization)
            assert result.horizon > 0
            assert result.window > 0
            assert result.system_identification.transient_time > 0
    
    def test_preprocessing_integration_with_real_config(self, configuration_file, data_directory, temp_result_directory):
        """Integration test with real configuration and data."""
        if not configuration_file.exists():
            pytest.skip("Configuration file not available")
        
        if not data_directory.exists():
            pytest.skip("Test data directory not available")
        
        # This would be a full integration test - skip if dependencies not available
        pytest.skip("Full integration test - requires complete environment setup")


class TestPreprocessingErrorHandling:
    """Test error handling in preprocessing."""
    
    def test_preprocessing_with_invalid_config(self):
        """Test preprocessing with invalid configuration."""
        with pytest.raises((FileNotFoundError, KeyError)):
            preprocess(
                "nonexistent_config.json",
                "dummy_dataset",
                "dummy_result",
                "test_model",
                "test_exp"
            )
    
    def test_preprocessing_with_invalid_dataset(self, temp_result_directory):
        """Test preprocessing with invalid dataset directory."""
        # This would require mocking configuration loading
        # Skip for now - would need more complex setup
        pytest.skip("Complex error handling test - requires full mock setup")


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing with real components."""
    
    def test_preprocessing_dataclass_consistency(self):
        """Test that preprocessing dataclasses work correctly."""
        # Test SystemIdentificationResults
        sys_id = SystemIdentificationResults(
            ss=None,  # Mock state space
            transient_time=100,
            fit=0.85,
            rmse=0.15
        )
        
        assert sys_id.transient_time == 100
        assert sys_id.fit == 0.85
        assert sys_id.rmse == 0.15
        
        # Test Normalization
        input_params = NormalizationParameters(
            mean=np.array([0.0]), 
            std=np.array([1.0])
        )
        output_params = NormalizationParameters(
            mean=np.array([0.0]), 
            std=np.array([1.0])
        )
        normalization = Normalization(input=input_params, output=output_params)
        
        # Test PreprocessingResults
        preprocessing = PreprocessingResults(
            system_identification=sys_id,
            horizon=100,
            window=10,
            normalization=normalization
        )
        
        assert preprocessing.horizon == 100
        assert preprocessing.window == 10
        assert preprocessing.system_identification.transient_time == 100


if __name__ == "__main__":
    pytest.main([__file__])
