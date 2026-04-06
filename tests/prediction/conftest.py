"""Fixtures for prediction writer tests."""

import pytest
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.preprocessing.preprocessing import (
    FEATURES_OUT_KEY,
    E_FIELD_OUT_KEY,
    H_FIELD_OUT_KEY,
    SUBJECT_OUT_KEY,
    COORDINATES_OUT_KEY,
    DTYPE_OUT_KEY,
    TRUNCATION_COEFFICIENTS_OUT_KEY,
    VOXEL_SIZE_OUT_KEY,
    MIN_EXTENT_OUT_KEY,
    MAX_EXTENT_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
    PROCESSED_ANTENNA_DIR_PATH,
    ANTENNA_MASKS_OUT_KEY,
)


@pytest.fixture
def grid_shape():
    """Standard grid shape for tests."""
    return (100, 100, 100)


@pytest.fixture
def num_coils():
    """Standard number of coils."""
    return 8


@pytest.fixture
def field_dtype():
    """Standard field dtype."""
    return np.float32


@pytest.fixture
def mock_grid_h5_file(tmp_path, grid_shape, field_dtype):
    """Create a mock grid H5 file for testing.

    Returns
    -------
    Path
        Path to created H5 file
    """
    # Create directory structure
    sim_dir = tmp_path / PROCESSED_SIMULATIONS_DIR_PATH
    sim_dir.mkdir(parents=True)

    # Create H5 file
    h5_path = sim_dir / "test_sim.h5"

    # Create structured dtype for fields
    complex_dtype = np.dtype([("re", field_dtype), ("im", field_dtype)])

    with h5py.File(h5_path, "w") as f:
        # Input features
        f.create_dataset(FEATURES_OUT_KEY, data=np.random.randn(3, *grid_shape).astype(field_dtype))

        # E-field and H-field with structured dtype
        efield = np.empty((8, 3, *grid_shape), dtype=complex_dtype)
        efield["re"] = np.random.randn(8, 3, *grid_shape).astype(field_dtype)
        efield["im"] = np.random.randn(8, 3, *grid_shape).astype(field_dtype)
        f.create_dataset(E_FIELD_OUT_KEY, data=efield)

        hfield = np.empty((8, 3, *grid_shape), dtype=complex_dtype)
        hfield["re"] = np.random.randn(8, 3, *grid_shape).astype(field_dtype)
        hfield["im"] = np.random.randn(8, 3, *grid_shape).astype(field_dtype)
        f.create_dataset(H_FIELD_OUT_KEY, data=hfield)

        # Subject mask
        f.create_dataset(SUBJECT_OUT_KEY, data=np.random.randint(0, 2, (6, *grid_shape), dtype=bool))

        # Positions
        f.create_dataset(COORDINATES_OUT_KEY, data=np.random.randn(3, *grid_shape).astype(field_dtype))

        # Metadata
        f.attrs[DTYPE_OUT_KEY] = field_dtype(1).dtype.name
        f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY] = np.array([1, 1, 1], dtype=field_dtype)
        f.attrs[VOXEL_SIZE_OUT_KEY] = 4
        f.attrs[MIN_EXTENT_OUT_KEY] = np.array([-200, -200, -200], dtype=field_dtype)
        f.attrs[MAX_EXTENT_OUT_KEY] = np.array([200, 200, 200], dtype=field_dtype)

    return h5_path


@pytest.fixture
def mock_source_dir(tmp_path, mock_grid_h5_file):
    """Create a mock source data directory with simulations.

    Returns
    -------
    Path
        Path to source directory
    """
    # The mock_grid_h5_file fixture already creates the directory structure
    return mock_grid_h5_file.parent.parent


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing.

    Returns
    -------
    torch.nn.Module
        Mock model that outputs shape (batch, 12, spatial...)
    """
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(5, 12, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    return MockModel()


@pytest.fixture
def mock_normalizer():
    """Create a mock normalizer for testing.

    Returns
    -------
    StandardNormalizer
        Mock normalizer that implements identity transforms
    """
    class MockNormalizer:
        def __call__(self, x):
            return x

        def inverse(self, x):
            return x

    return MockNormalizer()


@pytest.fixture
def mock_coil_predictions(grid_shape):
    """Create mock predictions for 8 coils.

    Returns
    -------
    List[np.ndarray]
        List of 8 predictions, each shape (12, *grid_shape)
    """
    predictions = []
    for i in range(8):
        pred = np.random.randn(12, *grid_shape).astype(np.float32)
        predictions.append(pred)
    return predictions


@pytest.fixture
def mock_batch(grid_shape):
    """Create a mock batch dictionary.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mock batch with required keys
    """
    return {
        'simulation': ['test_sim'],
        'input': torch.randn(1, 3, *grid_shape),
        'coils': torch.randn(1, 2, *grid_shape),
        'field': torch.randn(1, 2, 3, 3, *grid_shape),
        'subject': torch.randint(0, 2, (1, *grid_shape), dtype=torch.bool),
    }
