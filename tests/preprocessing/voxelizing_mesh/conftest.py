"""Fixtures for mesh voxelizer tests."""

import numpy as np
import pytest
from trimesh.primitives import Box, Sphere


@pytest.fixture(scope="module")
def sphere_unit_mesh():
    """Create a unit sphere mesh centered at the origin."""
    return Sphere(radius=1, center=(0, 0, 0))


@pytest.fixture(scope="module")
def box_unit_mesh():
    """Create a unit box mesh with bounds from (-1, -1, -1) to (1, 1, 1)."""
    bounds = np.array([[-1, -1, -1], [1, 1, 1]])
    return Box(bounds=bounds)
