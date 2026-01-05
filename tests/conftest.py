"""
Pytest configuration and fixtures for the test suite.
"""

import pytest
import torch
import numpy as np


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def device():
    """Provide device for tests"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
