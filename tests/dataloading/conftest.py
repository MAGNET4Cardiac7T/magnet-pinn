import pytest

import numpy as np

from magnet_pinn.data.dataitem import DataItem


@pytest.fixture
def zero_grid_item():
    return DataItem(
        simulation="",
        input=np.zeros((3, 20, 20, 20), dtype=np.float32),
        field=np.zeros((2, 2, 3, 20, 20, 20, 8), dtype=np.float32),
        subject=np.zeros((20, 20, 20), dtype=np.bool_),
        phase=np.zeros(8, dtype=np.float32),
        mask=np.zeros(8, dtype=np.bool_),
        coils=np.zeros((20, 20, 20, 8), dtype=np.float32),
        dtype="float32",
        truncation_coefficients=np.zeros(3, dtype=np.float32)
    )


@pytest.fixture
def random_grid_item():
    return DataItem(
        simulation="",
        input=np.random.rand(3, 20, 20, 20).astype(np.float32),
        field=np.random.rand(2, 2, 3, 20, 20, 20, 8).astype(np.float32),
        subject=np.random.choice([0, 1], size=(20, 20, 20)).astype(np.bool_),
        phase=np.random.rand(8).astype(np.float32),
        mask=np.random.choice([0, 1], size=8).astype(np.bool_),
        coils=np.random.choice([0, 1], size=(20, 20, 20, 8)).astype(np.float32),
        dtype="float32",
        truncation_coefficients=np.ones(3, dtype=np.float32)
    )


@pytest.fixture
def random_pointcloud_item():
    return DataItem(
        simulation="",
        input=np.random.rand(8000, 3).astype(np.float32),
        field=np.random.rand(2, 2, 3, 8000, 8).astype(np.float32),
        subject=np.random.choice([0, 1], size=8000).astype(np.bool_),
        phase=np.random.rand(8).astype(np.float32),
        mask=np.random.choice([0, 1], size=8).astype(np.bool_),
        coils=np.random.choice([0, 1], size=(8000, 8)).astype(np.float32),
        dtype="float32",
        truncation_coefficients=np.ones(3, dtype=np.float32)
    )
