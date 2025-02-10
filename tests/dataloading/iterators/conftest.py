from shutil import rmtree

import pytest
import numpy as np
from h5py import File

from tests.dataloading.iterators.helpers import (
    create_grid_processed_dir
)


GRID_RPOCESSED_DIR_NAME = "test_grid_voxel_size_4_data_type_float32"
GRID_PROCESSED_DIR_NAME_SHORT_TERM = "test_grid_voxel_size_4_data_type_float32_short_term"


@pytest.fixture(scope='module')
def grid_processed_dir(processed_dir_path, random_grid_item, zero_grid_item):
    grid_processed_dir_path = processed_dir_path / GRID_RPOCESSED_DIR_NAME
    create_grid_processed_dir(grid_processed_dir_path, random_grid_item, zero_grid_item)
    yield grid_processed_dir_path
    if grid_processed_dir_path.exists():
        rmtree(grid_processed_dir_path)


@pytest.fixture(scope='function')
def grid_prodcessed_dir_short_term(processed_dir_path, random_grid_item, zero_grid_item):
    grid_processed_dir_path = processed_dir_path / GRID_PROCESSED_DIR_NAME_SHORT_TERM
    create_grid_processed_dir(grid_processed_dir_path, random_grid_item, zero_grid_item)
    yield grid_processed_dir_path
    if grid_processed_dir_path.exists():
        rmtree(grid_processed_dir_path)
