import pytest
import numpy as np

from magnet_pinn.data.dataitem import DataItem
from tests.dataloading.helpers import (
    create_random_grid_item, create_random_pointcloud_item,
    create_zero_grid_item, create_zero_pointcloud_item
)


@pytest.fixture(scope="function")
def zero_grid_item():
    return create_zero_grid_item()


@pytest.fixture(scope="function")
def random_grid_item():
    return create_random_grid_item()


@pytest.fixture(scope="function")
def random_pointcloud_item():
    return create_random_pointcloud_item()


@pytest.fixture(scope="function")
def zero_pointcloud_item():
    return create_zero_pointcloud_item()
