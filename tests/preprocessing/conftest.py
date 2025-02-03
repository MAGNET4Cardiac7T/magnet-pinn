import pytest
from shutil import rmtree

from tests.preprocessing.helpers import (
    CENTRAL_BATCH_DIR_NAME, CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME,
    ANTENNA_SHORT_TERM_DIR_NAME, SHIFTED_SPHERE_SIM_NAME, SHIFTED_BOX_SIM_NAME, 
    CENTRAL_BATCH_SHORT_TERM_DIR_NAME,
    create_central_batch, create_shifted_batch, create_antenna_test_data
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
def pointslist_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    rmtree(simulation_path)
