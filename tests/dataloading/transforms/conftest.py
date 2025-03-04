import pytest

import numpy as np

from magnet_pinn.data.dataitem import DataItem


@pytest.fixture(scope="function")
def random_pointcloud_item_for_features_rearrange():
    return DataItem(
        simulation="",
        input=np.random.rand(8000, 3).astype(np.float32),
        field=np.random.rand(2, 2, 3, 8000).astype(np.float32),
        subject=np.random.choice([0, 1], size=8000).astype(np.bool_),
        positions=np.random.rand(8000, 3).astype(np.float32),
        phase=np.random.rand(8).astype(np.float32),
        mask=np.random.choice([0, 1], size=8).astype(np.bool_),
        coils=np.random.rand(2, 8000).astype(np.float32),
        dtype="float32",
        truncation_coefficients=np.ones(3, dtype=np.float32)
    )
