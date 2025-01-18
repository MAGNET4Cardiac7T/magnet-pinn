import pytest

import numpy as np

from magnet_pinn.data.dataitem import DataItem


@pytest.fixture
def zero_item():
    return DataItem(
        simulation="",
        field=np.zeros((2, 2, 3, 20, 20, 20, 8), dtype=np.float32),
        input=np.zeros((3, 20, 20, 20), dtype=np.float32),
        subject=np.zeros((20, 20, 20), dtype=np.int8),
        coils=np.zeros((20, 20, 20, 8), dtype=np.int8)
    )


@pytest.fixture
def random_item():
    return DataItem(
        simulation="",
        field=np.random.rand(2, 2, 3, 20, 20, 20, 8),
        input=np.random.rand(3, 20, 20, 20),
        subject=np.random.choice([0, 1], size=(20, 20, 20)),
        coils=np.random.choice([0, 1], size=(20, 20, 20, 8))
    )
