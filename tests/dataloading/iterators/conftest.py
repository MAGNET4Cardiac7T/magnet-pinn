from shutil import rmtree

import pytest
import numpy as np
from h5py import File

from magnet_pinn.data.transforms import (
    Compose, PhaseShift, PointPhaseShift
)
from tests.dataloading.iterators.helpers import (
    create_processed_dir
)


GRID_RPOCESSED_DIR_NAME = "test_grid_voxel_size_4_data_type_float32"
GRID_PROCESSED_DIR_NAME_SHORT_TERM = "test_grid_voxel_size_4_data_type_float32_short_term"
POINTCLOUD_PREPROCESSED_DIR_NAME = "test_point_data_type_float32"
POINTCLOUD_PREPROCESSED_DIR_NAME_SHORT_TERM = "test_point_data_type_float32_short_term"


@pytest.fixture(scope='module')
def grid_processed_dir(processed_dir_path, random_grid_item, zero_grid_item):
    grid_processed_dir_path = processed_dir_path / GRID_RPOCESSED_DIR_NAME
    create_processed_dir(grid_processed_dir_path, random_grid_item, zero_grid_item, is_grid=True)
    yield grid_processed_dir_path
    if grid_processed_dir_path.exists():
        rmtree(grid_processed_dir_path)


@pytest.fixture(scope='function')
def grid_prodcessed_dir_short_term(processed_dir_path, random_grid_item, zero_grid_item):
    grid_processed_dir_path = processed_dir_path / GRID_PROCESSED_DIR_NAME_SHORT_TERM
    create_processed_dir(grid_processed_dir_path, random_grid_item, zero_grid_item, is_grid=True)
    yield grid_processed_dir_path
    if grid_processed_dir_path.exists():
        rmtree(grid_processed_dir_path)


@pytest.fixture(scope='module')
def pointcloud_processed_dir(processed_dir_path, random_pointcloud_item, zero_pointcloud_item):
    pointcloud_processed_dir_path = processed_dir_path / POINTCLOUD_PREPROCESSED_DIR_NAME
    create_processed_dir(pointcloud_processed_dir_path, random_pointcloud_item, zero_pointcloud_item, is_grid=False)
    yield pointcloud_processed_dir_path
    if pointcloud_processed_dir_path.exists():
        rmtree(pointcloud_processed_dir_path)


@pytest.fixture(scope='function')
def pointcloud_processed_dir_short_term(processed_dir_path, random_pointcloud_item, zero_pointcloud_item):
    pointcloud_processed_dir_path = processed_dir_path / POINTCLOUD_PREPROCESSED_DIR_NAME_SHORT_TERM
    create_processed_dir(pointcloud_processed_dir_path, random_pointcloud_item, zero_pointcloud_item, is_grid=False)
    yield pointcloud_processed_dir_path
    if pointcloud_processed_dir_path.exists():
        rmtree(pointcloud_processed_dir_path)


@pytest.fixture(scope='module')
def grid_aug():
    return PhaseShift(num_coils=8)


@pytest.fixture(scope='module')
def pointcloud_aug():
    return PointPhaseShift(num_coils=8)
