"""Pytest fixtures for dataloading tests.

Provides test fixtures for grid and pointcloud DataItem objects with various
initialization patterns (zero-filled and random data) to support transformation
and dataloading test scenarios.
"""

import pytest
import numpy as np

from magnet_pinn.data.dataitem import DataItem


@pytest.fixture(scope="module")
def zero_grid_item():
    """Create a DataItem fixture with zero-filled grid data.

    Returns a DataItem with 3D grid structure (20x20x20) where all arrays are
    initialized to zeros. Useful for testing transformation behavior on neutral data.

    Returns
    -------
    DataItem
        Grid-based DataItem with zero-initialized fields.
    """
    return DataItem(
        simulation="children_0_tubes_0_id_1",
        input=np.zeros((3, 20, 20, 20), dtype=np.float32),
        field=np.zeros((2, 2, 8, 3, 20, 20, 20), dtype=np.float32),
        subject=np.zeros((1, 20, 20, 20), dtype=np.bool_),
        positions=np.zeros((3, 20, 20, 20), dtype=np.float32),
        phase=np.zeros(8, dtype=np.float32),
        mask=np.zeros(8, dtype=np.bool_),
        coils=np.zeros((8, 20, 20, 20), dtype=np.float32),
        dtype="float32",
        truncation_coefficients=np.zeros(3, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def random_grid_item():
    """Create a DataItem fixture with random grid data.

    Returns a DataItem with 3D grid structure (20x20x20) where arrays are filled
    with random values. Useful for testing transformations on realistic data patterns.

    Returns
    -------
    DataItem
        Grid-based DataItem with randomly initialized fields.
    """
    return DataItem(
        simulation="children_0_tubes_0_id_0",
        input=np.random.rand(3, 20, 20, 20).astype(np.float32),
        field=np.random.rand(2, 2, 8, 3, 20, 20, 20).astype(np.float32),
        subject=np.random.choice([0, 1], size=(1, 20, 20, 20)).astype(np.bool_),
        positions=np.random.rand(3, 20, 20, 20).astype(np.float32),
        phase=np.random.rand(8).astype(np.float32),
        mask=np.random.choice([0, 1], size=8).astype(np.bool_),
        coils=np.random.choice([0, 1], size=(8, 20, 20, 20)).astype(np.float32),
        dtype="float32",
        truncation_coefficients=np.ones(3, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def random_pointcloud_item():
    """Create a DataItem fixture with random pointcloud data.

    Returns a DataItem with pointcloud structure (8000 points) where arrays are filled
    with random values. Useful for testing transformations on point-based data.

    Returns
    -------
    DataItem
        Pointcloud-based DataItem with randomly initialized fields.
    """
    return DataItem(
        simulation="children_0_tubes_0_id_0",
        input=np.random.rand(3, 8000).astype(np.float32),
        field=np.random.rand(2, 2, 8, 3, 8000).astype(np.float32),
        subject=np.random.choice([0, 1], size=(1, 8000)).astype(np.bool_),
        positions=np.random.rand(3, 8000).astype(np.float32),
        phase=np.random.rand(8).astype(np.float32),
        mask=np.random.choice([0, 1], size=8).astype(np.bool_),
        coils=np.random.choice([0, 1], size=(8, 8000)).astype(np.float32),
        dtype="float32",
        truncation_coefficients=np.ones(3, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def zero_pointcloud_item():
    """Create a DataItem fixture with zero-filled pointcloud data.

    Returns a DataItem with pointcloud structure (8000 points) where all arrays are
    initialized to zeros. Useful for testing transformation behavior on neutral point data.

    Returns
    -------
    DataItem
        Pointcloud-based DataItem with zero-initialized fields.
    """
    return DataItem(
        simulation="children_0_tubes_0_id_1",
        input=np.zeros((3, 8000), dtype=np.float32),
        field=np.zeros((2, 2, 8, 3, 8000), dtype=np.float32),
        subject=np.zeros((1, 8000), dtype=np.bool_),
        positions=np.zeros((3, 8000), dtype=np.float32),
        phase=np.zeros(8, dtype=np.float32),
        mask=np.zeros(8, dtype=np.bool_),
        coils=np.zeros((8, 8000), dtype=np.float32),
        dtype="float32",
        truncation_coefficients=np.zeros(3, dtype=np.float32),
    )
