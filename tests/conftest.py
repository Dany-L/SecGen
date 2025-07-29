"""Pytest configuration and fixtures for CRNN tests."""
import os
import sys
import pytest
import torch
import tempfile
import shutil
import warnings
from pathlib import Path

# Add the src directory to the Python path
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Set torch default dtype
torch.set_default_dtype(torch.double)

# Filter out Google protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google._upb._message")
warnings.filterwarnings("ignore", message=".*google._upb._message.*PyType_Spec.*", category=DeprecationWarning)


@pytest.fixture(scope="session")
def test_root_dir():
    """Root directory for tests."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def configuration_file(test_root_dir):
    """Path to the test configuration file."""
    return test_root_dir / "config" / "config.json"


@pytest.fixture(scope="session")
def data_directory(test_root_dir):
    """Path to the test data directory."""
    return test_root_dir / "data"


@pytest.fixture(scope="function")
def temp_result_directory():
    """Temporary result directory that gets cleaned up after each test."""
    temp_dir = tempfile.mkdtemp(prefix="crnn_test_results_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def session_result_directory(test_root_dir):
    """Persistent result directory for the entire test session."""
    result_dir = test_root_dir / "_tmp" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


@pytest.fixture(autouse=True)
def setup_torch():
    """Ensure torch is set up correctly for each test."""
    torch.set_default_dtype(torch.double)
    # Reset random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "smoke: mark test as a smoke test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add smoke marker to smoke_test.py tests
        if "smoke_test" in item.nodeid:
            item.add_marker(pytest.mark.smoke)
        
        # Add slow marker to tests that likely take longer
        if any(keyword in item.name.lower() for keyword in ["training", "evaluation", "continue"]):
            item.add_marker(pytest.mark.slow)
