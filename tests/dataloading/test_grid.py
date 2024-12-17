import pytest
import numpy as np
from pathlib import Path
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.dataitem import DataItem

@pytest.fixture
def mock_magnet_grid_iterator():
    class MockMagnetGridIterator(MagnetGridIterator):
        def _read_input(self, simulation_path):
            return np.array([1, 2, 3])

        def _read_subject(self, simulation_path):
            return "subject"

        def _get_simulation_name(self, simulation_path):
            return "simulation_name"

        def _read_fields(self, simulation_path):
            return np.array([4, 5, 6])

        def _get_dtype(self, simulation_path):
            return np.float32

        def _get_truncation_coefficients(self, simulation_path):
            return np.array([0.1, 0.2, 0.3])

    return MockMagnetGridIterator(num_coils=3, coils=[1, 2, 3])

def test_load_simulation(mock_magnet_grid_iterator):
    simulation_path = "dummy_path"
    data_item = mock_magnet_grid_iterator._load_simulation(simulation_path)

    assert isinstance(data_item, DataItem)
    assert np.array_equal(data_item.input, np.array([1, 2, 3]))
    assert data_item.subject == "subject"
    assert data_item.simulation == "simulation_name"
    assert np.array_equal(data_item.field, np.array([4, 5, 6]))
    assert np.array_equal(data_item.phase, np.zeros(3))
    assert np.array_equal(data_item.mask, np.ones(3))
    assert data_item.coils == [1, 2, 3]
    assert data_item.dtype == np.float32
    assert np.array_equal(data_item.truncation_coefficients, np.array([0.1, 0.2, 0.3]))
    