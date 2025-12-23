"""Pytest fixtures for preprocessing module tests."""

from pathlib import Path
from shutil import rmtree
from typing import Generator

import pytest

from tests.preprocessing.helpers import (
    ANTENNA_SHORT_TERM_DIR_NAME,
    CENTRAL_BATCH_DIR_NAME,
    CENTRAL_BATCH_SHORT_TERM_DIR_NAME,
    CENTRAL_BOX_SIM_NAME,
    CENTRAL_SPHERE_SIM_NAME,
    SHIFTED_BOX_SIM_NAME,
    SHIFTED_SPHERE_SIM_NAME,
    create_antenna_test_data,
    create_central_batch,
    create_duplicate_batch,
    create_shifted_batch,
)

ALL_SIM_NAMES = [
    CENTRAL_SPHERE_SIM_NAME,
    CENTRAL_BOX_SIM_NAME,
    SHIFTED_SPHERE_SIM_NAME,
    SHIFTED_BOX_SIM_NAME,
]


@pytest.fixture(scope="function")
def processed_batch_dir_path(processed_dir_path: Path) -> Generator[Path, None, None]:
    """Create and yield a processed batch directory path, clean up after test."""
    batch_path = processed_dir_path / CENTRAL_BATCH_DIR_NAME
    batch_path.mkdir(parents=True, exist_ok=True)
    yield batch_path
    if batch_path.exists():
        rmtree(batch_path)


@pytest.fixture(scope="module")
def raw_central_batch_dir_path(data_dir_path: Path) -> Generator[Path, None, None]:
    """Create and yield a raw central batch directory with simulation data."""
    batch_dir_path = create_central_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture(scope="module")
def raw_shifted_batch_dir_path(data_dir_path: Path) -> Generator[Path, None, None]:
    """Create and yield a raw shifted batch directory with simulation data."""
    batch_dir_path = create_shifted_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture(scope="function")
def raw_central_batch_short_term(data_dir_path: Path) -> Generator[Path, None, None]:
    """Create and yield a short-term raw central batch directory."""
    batch_dir_path = create_central_batch(
        data_dir_path, CENTRAL_BATCH_SHORT_TERM_DIR_NAME
    )
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture(scope="function")
def raw_antenna_dir_path_short_term(
    data_dir_path: Path,
) -> Generator[Path, None, None]:
    """Create and yield a short-term antenna directory with test data."""
    antenna_path = create_antenna_test_data(data_dir_path, ANTENNA_SHORT_TERM_DIR_NAME)
    yield antenna_path
    if antenna_path.exists():
        rmtree(antenna_path)


@pytest.fixture(scope="module")
def raw_antenna_dir_path(data_dir_path: Path) -> Generator[Path, None, None]:
    """Create and yield an antenna directory with test data."""
    antenna_path = create_antenna_test_data(data_dir_path)
    yield antenna_path
    if antenna_path.exists():
        rmtree(antenna_path)


@pytest.fixture(scope="module")
def grid_simulation_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Create and yield a temporary grid simulation path."""
    simulation_path = tmp_path_factory.mktemp("simulation_name")
    yield simulation_path
    if simulation_path.exists():
        rmtree(simulation_path)


@pytest.fixture(scope="module")
def pointslist_simulation_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Create and yield a temporary pointslist simulation path."""
    simulation_path = tmp_path_factory.mktemp("simulation_name")
    yield simulation_path
    if simulation_path.exists():
        rmtree(simulation_path)


@pytest.fixture(scope="module")
def raw_duplicate_batch_dir_path(data_dir_path: Path) -> Generator[Path, None, None]:
    """Create and yield a raw duplicate batch directory with simulation data."""
    batch_dir_path = create_duplicate_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)
