import pytest
import numpy as np
import trimesh
from shutil import rmtree

from magnet_pinn.generator.typing import MeshPhantom, PropertyPhantom, PropertyItem


@pytest.fixture(scope='module')
def generation_dir_path(tmp_path_factory):
    generation_path = tmp_path_factory.mktemp('generation')
    yield generation_path
    if generation_path.exists():
        rmtree(generation_path)


@pytest.fixture(scope='function')
def generation_output_dir_path(generation_dir_path):
    output_dir = generation_dir_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    if output_dir.exists():
        rmtree(output_dir)


@pytest.fixture
def simple_mesh():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def property_item():
    return PropertyItem(
        conductivity=0.5,
        permittivity=80.0,
        density=1000.0
    )


@pytest.fixture
def mesh_phantom(simple_mesh):
    return MeshPhantom(
        parent=simple_mesh,
        children=[simple_mesh, simple_mesh],
        tubes=[simple_mesh]
    )


@pytest.fixture
def property_phantom(property_item):
    parent = PropertyItem(conductivity=0.1, permittivity=10.0, density=100.0)
    children = [
        PropertyItem(conductivity=0.2, permittivity=20.0, density=200.0),
        PropertyItem(conductivity=0.3, permittivity=30.0, density=300.0)
    ]
    tubes = [PropertyItem(conductivity=0.8, permittivity=80.0, density=800.0)]
    
    return PropertyPhantom(parent=parent, children=children, tubes=tubes)
