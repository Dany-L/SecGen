# CRNN Tests

This directory contains the test suite for the CRNN package.

## Test Structure

### Smoke Tests (`smoke_test.py`)

The smoke tests verify that all model configurations can be trained, continued, and evaluated without errors. Each model/experiment/device combination is tested as a separate pytest test case.

**Available models:**
- `ltiRnn`: Basic LTI RNN with tanh activation
- `lstm`: Basic LSTM with dropout
- `rnn`: Basic RNN with multiple layers
- `tanh`: Constrained LTI RNN with tanh nonlinearity
- `dzn`: Constrained LTI RNN with deadzone nonlinearity
- `dznGen`: Constrained LTI RNN with general sector conditions

## Running Tests

### Run all smoke tests
```bash
pytest tests/smoke_test.py
```

### Run tests for specific models
```bash
pytest tests/smoke_test.py -k "lstm"
pytest tests/smoke_test.py -k "tanh"
```

### Run tests excluding slow ones
```bash
pytest tests/smoke_test.py -m "not slow"
```

### Run only configuration and data tests (fast)
```bash
pytest tests/smoke_test.py -k "not training_and_evaluation"
```

### Run with parallel execution (if pytest-xdist is installed)
```bash
pytest tests/smoke_test.py -n auto
```

### Verbose output with test details
```bash
pytest tests/smoke_test.py -v -s
```

## Test Configuration

The tests use the configuration file `tests/config/config.json` which defines:
- Model parameters and classes
- Experiment configurations
- Training parameters (epochs, batch size, etc.)
- Input/output specifications

## Test Data

Tests expect data to be available in `tests/data/` directory. The specific data files required depend on the experiment configurations.

## Backwards Compatibility

The smoke test can still be run directly (without pytest) for backwards compatibility:
```bash
python tests/smoke_test.py
```

However, using pytest is recommended for better test isolation and reporting.

## Test Markers

- `@pytest.mark.smoke`: Marks smoke tests
- `@pytest.mark.slow`: Marks slow-running tests
- `@pytest.mark.integration`: Marks integration tests

## Fixtures

- `configuration_file`: Path to test configuration
- `data_directory`: Path to test data
- `temp_result_directory`: Temporary directory for test results (cleaned up after each test)
- `session_result_directory`: Persistent result directory for debugging
