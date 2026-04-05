"""Pytest configuration and shared fixtures for the test suite.

This module provides common test fixtures including deterministic random seeds,
temporary directory management, and cleanup utilities for the entire test suite.
"""

import random
from shutil import rmtree
from pathlib import Path

import pytest
import numpy as np


PROCESSED_DIR_PATH = "processed"


@pytest.fixture(autouse=True)
def deterministicity():
    """Set random seeds for deterministic test execution.

    This fixture automatically applies to all tests, ensuring reproducible
    results by setting a fixed seed (42) for both NumPy and Python's random
    number generators.
    """
    seed = 42
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture(scope="session", autouse=True)
def cleanup_basetemp(request):
    """Clean up the basetemp directory after all tests complete."""
    yield
    # Clean up basetemp if it was explicitly set via --basetemp flag
    basetemp = request.config.option.basetemp
    if basetemp:
        basetemp_path = Path(basetemp)
        if basetemp_path.exists():
            rmtree(basetemp_path, ignore_errors=True)


@pytest.fixture(scope="module")
def data_dir_path(tmp_path_factory):
    """Create a temporary data directory for module-scoped tests.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest's temporary path factory for creating test directories.

    Yields
    ------
    pathlib.Path
        Path to the temporary data directory.
    """
    data_path = tmp_path_factory.mktemp("data")
    yield data_path
    if data_path.exists():
        rmtree(data_path)


@pytest.fixture(scope="module")
def processed_dir_path(data_dir_path):
    """Create a processed data subdirectory within the data directory.

    Parameters
    ----------
    data_dir_path : pathlib.Path
        Parent data directory path from the data_dir_path fixture.

    Yields
    ------
    pathlib.Path
        Path to the temporary processed data subdirectory.
    """
    processed_dir = data_dir_path / PROCESSED_DIR_PATH
    processed_dir.mkdir()
    yield processed_dir
    if processed_dir.exists():
        rmtree(processed_dir)
