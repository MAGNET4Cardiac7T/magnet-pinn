import pytest
from shutil import rmtree

import numpy as np

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    H_FIELD_DATABASE_KEY
)
from tests.preprocessing.helpers import (
    CENTRAL_BATCH_DIR_NAME, CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME,
    ANTENNA_SHORT_TERM_DIR_NAME, SHIFTED_SPHERE_SIM_NAME, SHIFTED_BOX_SIM_NAME, 
    CENTRAL_BATCH_SHORT_TERM_DIR_NAME,
    create_central_batch, create_shifted_batch, create_antenna_test_data,
    create_grid_field, create_grid_field_with_mixed_axis_order, create_pointslist_field
)


ALL_SIM_NAMES = [
    CENTRAL_SPHERE_SIM_NAME,
    CENTRAL_BOX_SIM_NAME,
    SHIFTED_SPHERE_SIM_NAME,
    SHIFTED_BOX_SIM_NAME
]


@pytest.fixture(scope="function")
def processed_batch_dir_path(processed_dir_path):
    batch_path = processed_dir_path / CENTRAL_BATCH_DIR_NAME
    batch_path.mkdir(parents=True, exist_ok=True)
    yield batch_path
    if batch_path.exists():
        rmtree(batch_path)


@pytest.fixture(scope='module')
def raw_central_batch_dir_path(data_dir_path):
    batch_dir_path = create_central_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture(scope='module')
def raw_shifted_batch_dir_path(data_dir_path):
    batch_dir_path = create_shifted_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture(scope="function")
def raw_central_batch_short_term(data_dir_path):
    batch_dir_path = create_central_batch(data_dir_path, CENTRAL_BATCH_SHORT_TERM_DIR_NAME)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture(scope='function')
def raw_antenna_dir_path_short_term(data_dir_path):
    antenna_path = create_antenna_test_data(data_dir_path, ANTENNA_SHORT_TERM_DIR_NAME)
    yield antenna_path
    if antenna_path.exists():
        rmtree(antenna_path)


@pytest.fixture(scope='module')
def raw_antenna_dir_path(data_dir_path):
    antenna_path = create_antenna_test_data(data_dir_path)
    yield antenna_path
    if antenna_path.exists():
        rmtree(antenna_path)


@pytest.fixture(scope='module')
def grid_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    rmtree(simulation_path)


@pytest.fixture(scope='module')
def e_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-240, -220, -250], [240, 220, 250]])

    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0
    )

    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0
    )

    return grid_simulation_path


@pytest.fixture(scope='module')
def h_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-240, -220, -250], [240, 220, 250]])

    create_grid_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0
    )

    create_grid_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds,
        0
    )

    return grid_simulation_path


@pytest.fixture(scope='module')
def e_field_grid_data_with_mixed_axis(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 121)
    )

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (126, 121, 111)
    )

    return grid_simulation_path


@pytest.fixture(scope='module')
def h_field_grid_data_with_mixed_axis(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 121)
    )

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (126, 121, 111)
    )

    return grid_simulation_path


@pytest.fixture(scope='module')
def e_field_grid_data_with_inconsistent_shape(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 126)
    )

    return grid_simulation_path


@pytest.fixture(scope='module')
def h_field_grid_data_with_inconsistent_shape(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 126)
    )

    return grid_simulation_path


@pytest.fixture(scope='module')
def pointslist_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    rmtree(simulation_path)


@pytest.fixture(scope='module')
def e_field_pointslist_data(pointslist_simulation_path):
    field_path = pointslist_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY
    )

    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY
    )
    return pointslist_simulation_path


@pytest.fixture(scope='module')
def h_field_pointslist_data(pointslist_simulation_path):
    field_path = pointslist_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_pointslist_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY    
    )

    create_pointslist_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY
    )

    return pointslist_simulation_path
