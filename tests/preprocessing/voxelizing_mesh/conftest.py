import pytest
import numpy as np
from trimesh.primitives import Sphere, Box


@pytest.fixture(scope='module')
def sphere_unit_mesh():
    return Sphere(radius=1, center=(0, 0, 0))


@pytest.fixture(scope='module')
def box_unit_mesh():
    bounds = np.array([
        [-1, -1, -1],
        [1, 1, 1]
    ])
    return Box(bounds=bounds
)
