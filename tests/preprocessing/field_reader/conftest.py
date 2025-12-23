"""Fixtures for field reader tests."""

import pytest
from shutil import rmtree

import numpy as np

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    H_FIELD_DATABASE_KEY,
)
from tests.preprocessing.helpers import (
    create_grid_field,
    create_grid_field_with_mixed_axis_order,
    create_pointslist_field,
)


@pytest.fixture(scope="module")
def e_field_grid_data(grid_simulation_path):
    """Create E-field grid test data."""
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-240, -220, -250], [240, 220, 250]])

    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0,
    )

    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0,
    )

    yield grid_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def h_field_grid_data(grid_simulation_path):
    """Create H-field grid test data."""
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-240, -220, -250], [240, 220, 250]])

    create_grid_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0,
    )

    create_grid_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0,
    )

    yield grid_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def e_field_grid_data_with_mixed_axis(grid_simulation_path):
    """Create E-field grid test data with mixed axis order."""
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 121),
    )

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (126, 121, 111),
    )

    yield grid_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def h_field_grid_data_with_mixed_axis(grid_simulation_path):
    """Create H-field grid test data with mixed axis order."""
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 121),
    )

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (126, 121, 111),
    )

    yield grid_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def e_field_grid_data_with_inconsistent_shape(grid_simulation_path):
    """Create E-field grid test data with inconsistent shape."""
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 126),
    )

    yield grid_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def h_field_grid_data_with_inconsistent_shape(grid_simulation_path):
    """Create H-field grid test data with inconsistent shape."""
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 126),
    )

    yield grid_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def e_field_pointslist_data(pointslist_simulation_path):
    """Create E-field pointslist test data."""
    field_path = pointslist_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC1].h5", E_FIELD_DATABASE_KEY
    )

    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC2].h5", E_FIELD_DATABASE_KEY
    )
    yield pointslist_simulation_path
    if field_path.exists():
        rmtree(field_path)


@pytest.fixture(scope="module")
def h_field_pointslist_data(pointslist_simulation_path):
    """Create H-field pointslist test data."""
    field_path = pointslist_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_pointslist_field(
        field_path / "h-field (f=297.2) [AC1].h5", H_FIELD_DATABASE_KEY
    )

    create_pointslist_field(
        field_path / "h-field (f=297.2) [AC2].h5", H_FIELD_DATABASE_KEY
    )

    yield pointslist_simulation_path
    if field_path.exists():
        rmtree(field_path)
