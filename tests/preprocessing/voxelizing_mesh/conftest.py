import pytest
from trimesh.primitives import Sphere


@pytest.fixture(scope='session')
def sphere_unit_mesh():
    return Sphere(radius=1, center=(0, 0, 0))
