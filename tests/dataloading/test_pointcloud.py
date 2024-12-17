import pytest
import numpy as np
from pathlib import Path
from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.preprocessing.preprocessing import COORDINATES_OUT_KEY
import h5py

@pytest.fixture
def mock_h5_file(tmp_path):
    file_path = tmp_path / "test_simulation.h5"
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(COORDINATES_OUT_KEY, data=np.random.rand(10, 3))
    return file_path

@pytest.fixture
def magnet_point_iterator():
    return MagnetPointIterator(num_coils=5, coils=[1, 2, 3, 4, 5])

def test_load_simulation(magnet_point_iterator, mock_h5_file):
    data_item = magnet_point_iterator._load_simulation(mock_h5_file)
    assert isinstance(data_item, DataItem)
    assert data_item.positions.shape == (10, 3)
    assert np.array_equal(data_item.phase, np.zeros(5))
    assert np.array_equal(data_item.mask, np.ones(5))
    assert data_item.coils == [1, 2, 3, 4, 5]

def test_read_positions(magnet_point_iterator, mock_h5_file):
    positions = magnet_point_iterator._read_positions(mock_h5_file)
    assert positions.shape == (10, 3)
