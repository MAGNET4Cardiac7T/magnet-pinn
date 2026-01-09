"""Test fixtures for generator tests.

Provides reusable test fixtures for phantom generation, mesh creation,
and property configuration used across generation test modules.
"""
from unittest.mock import patch

import numpy as np
import pytest
import trimesh
from shutil import rmtree

from magnet_pinn.generator.structures import Blob, CustomMeshStructure
from magnet_pinn.generator.typing import MeshPhantom, PropertyItem, PropertyPhantom
from magnet_pinn.utils import PerlinNoise


def _fast_blob_init(
    self,
    position: np.ndarray,
    radius: float,
    num_octaves: int = 3,
    relative_disruption_strength: float = 0.1,
    seed: int = 42,
    perlin_scale: float = 0.4,
):
    """Fast Blob initialization that skips expensive Fibonacci sampling.

    This replacement __init__ sets reasonable dummy values for empirical
    offset attributes instead of computing them from 10,000 sample points.
    Used by test fixtures to speed up generation tests.

    The empirical offset values are based on typical Perlin noise behavior:
    - Max offset is roughly +0.25 * relative_disruption_strength
    - Min offset is roughly -0.25 * relative_disruption_strength
    These values maintain compatibility with packing algorithms that use
    these offsets for collision detection and margin calculations.
    """
    from magnet_pinn.generator.structures import Structure3D
    Structure3D.__init__(self, position=position, radius=radius)

    self.relative_disruption_strength = relative_disruption_strength

    if perlin_scale == 0:
        raise ValueError(
            "perlin_scale cannot be zero as it causes division by zero in offset calculations."
        )
    self.perlin_scale = perlin_scale

    self.noise = PerlinNoise(octaves=num_octaves, seed=seed)
    self.empirical_max_offset = 0.025 * (relative_disruption_strength / 0.1)
    self.empirical_min_offset = -0.025 * (relative_disruption_strength / 0.1)
    self.effective_radius = self.radius * (1 + self.empirical_max_offset)


@pytest.fixture(autouse=True, scope="module")
def fast_blob_initialization():
    """Speed up tests by patching Blob to skip expensive initialization.

    Blob.__init__ normally generates 10,000 Fibonacci sphere points and
    calculates Perlin noise offsets for each, which takes ~0.35s per Blob.
    This fixture patches Blob.__init__ to use dummy offset values instead,
    dramatically speeding up tests that create multiple Blobs.
    """
    with patch.object(Blob, "__init__", _fast_blob_init):
        yield


@pytest.fixture(scope='module')
def generation_dir_path(tmp_path_factory):
    """Provide a module-scoped temporary directory for generation tests.

    Creates a temporary directory that persists for the entire test module
    and is automatically cleaned up after all tests complete.
    """
    generation_path = tmp_path_factory.mktemp('generation')
    yield generation_path
    if generation_path.exists():
        rmtree(generation_path)


@pytest.fixture(scope='function')
def generation_output_dir_path(generation_dir_path):
    """Provide a function-scoped output directory for each test.

    Creates a fresh output subdirectory for each test function and cleans
    it up after the test completes.
    """
    output_dir = generation_dir_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    if output_dir.exists():
        rmtree(output_dir)


@pytest.fixture
def simple_mesh():
    """Provide a simple triangular mesh for testing.

    Creates a minimal valid trimesh object with 3 vertices forming a single
    triangular face in the XY plane.
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def property_item():
    """Provide a standard PropertyItem with typical material properties.

    Creates a PropertyItem with realistic tissue-like properties for testing.
    """
    return PropertyItem(
        conductivity=0.5,
        permittivity=80.0,
        density=1000.0
    )


@pytest.fixture
def mesh_phantom(simple_mesh):
    """Provide a MeshPhantom with parent and child meshes.

    Creates a test phantom containing a parent mesh and two identical children
    plus a tube mesh for testing hierarchical mesh structures.
    """
    return MeshPhantom(
        parent=simple_mesh,
        children=[simple_mesh, simple_mesh],
        tubes=[simple_mesh]
    )


@pytest.fixture
def property_phantom(property_item):
    """Provide a PropertyPhantom with varied material properties.

    Creates a test phantom with distinct property values for parent, children,
    and tubes to verify property handling across phantom hierarchy.
    """
    parent = PropertyItem(conductivity=0.1, permittivity=10.0, density=100.0)
    children = [
        PropertyItem(conductivity=0.2, permittivity=20.0, density=200.0),
        PropertyItem(conductivity=0.3, permittivity=30.0, density=300.0)
    ]
    tubes = [PropertyItem(conductivity=0.8, permittivity=80.0, density=800.0)]

    return PropertyPhantom(parent=parent, children=children, tubes=tubes)


@pytest.fixture(scope='module')
def custom_mesh_stl_file(tmp_path_factory):
    """Provide a module-scoped STL file path for CustomMeshStructure tests.

    Creates a simple box mesh (2x2x2 cube) and exports it to an STL file
    that persists for the entire test module. This avoids repeated file I/O
    for tests that use CustomMeshStructure.
    """
    mesh_dir = tmp_path_factory.mktemp('custom_mesh')
    stl_path = mesh_dir / 'parent.stl'
    box_mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    box_mesh.export(str(stl_path))
    yield stl_path


@pytest.fixture(scope='module')
def custom_mesh_structure(custom_mesh_stl_file):
    """Provide a module-scoped CustomMeshStructure for tests.

    Creates a CustomMeshStructure from the shared STL file. The mesh is
    loaded once per module and reused across tests. Tests should NOT
    mutate this structure.
    """
    mesh_structure = CustomMeshStructure(str(custom_mesh_stl_file))
    mesh_structure.mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    return mesh_structure
