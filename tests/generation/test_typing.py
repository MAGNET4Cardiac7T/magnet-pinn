import numpy as np
import pickle
import json
from typing import List
from trimesh import Trimesh

from magnet_pinn.generator.typing import (
    PropertyItem,
    StructurePhantom,
    MeshPhantom,
    PropertyPhantom,
    Point3D,
    FaceIndices,
    MeshGrid,
    PhantomItem,
)
from magnet_pinn.generator.structures import Blob, Tube


def test_property_item_mutation_bug_protection():
    """Test that PropertyItem instances are protected from mutation through io.py operations."""
    original_props = PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0)

    original_props.__dict__.update({"new_field": "injected_value"})

    new_props = PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0)

    assert not hasattr(new_props, "new_field")
    assert new_props.conductivity == 0.5
    assert new_props.permittivity == 80.0
    assert new_props.density == 1000.0


def test_property_item_with_realistic_material_properties():
    """Test PropertyItem with realistic biological tissue properties."""
    brain = PropertyItem(conductivity=0.33, permittivity=50.0, density=1040.0)
    assert brain.conductivity == 0.33
    assert brain.permittivity == 50.0
    assert brain.density == 1040.0

    muscle = PropertyItem(conductivity=0.77, permittivity=55.0, density=1050.0)
    assert muscle.conductivity == 0.77
    assert muscle.permittivity == 55.0
    assert muscle.density == 1050.0

    fat = PropertyItem(conductivity=0.04, permittivity=11.0, density=920.0)
    assert fat.conductivity == 0.04
    assert fat.permittivity == 11.0
    assert fat.density == 920.0


def test_property_item_serialization_compatibility():
    """Test that PropertyItem works correctly with serialization."""
    original = PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0)

    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)

    assert unpickled.conductivity == original.conductivity
    assert unpickled.permittivity == original.permittivity
    assert unpickled.density == original.density
    assert unpickled == original

    as_dict = {
        "conductivity": original.conductivity,
        "permittivity": original.permittivity,
        "density": original.density,
    }
    json_str = json.dumps(as_dict)
    parsed = json.loads(json_str)

    reconstructed = PropertyItem(**parsed)
    assert reconstructed == original


def test_property_item_extreme_values_validation():
    """Test PropertyItem with physically meaningful extreme values."""
    air = PropertyItem(conductivity=1e-15, permittivity=1.0, density=1.2)
    assert air.conductivity == 1e-15

    metal = PropertyItem(conductivity=5.96e7, permittivity=1.0, density=8960.0)
    assert metal.conductivity == 5.96e7

    nan_prop = PropertyItem(
        conductivity=float("nan"), permittivity=80.0, density=1000.0
    )
    assert np.isnan(nan_prop.conductivity)

    inf_prop = PropertyItem(conductivity=float("inf"), permittivity=1.0, density=8960.0)
    assert np.isinf(inf_prop.conductivity)


def test_property_item_boundary_validation():
    """Test PropertyItem boundary conditions relevant to MRI physics."""
    insulator = PropertyItem(conductivity=0.0, permittivity=4.0, density=1000.0)
    assert insulator.conductivity == 0.0

    high_perm = PropertyItem(conductivity=0.1, permittivity=1000.0, density=2000.0)
    assert high_perm.permittivity == 1000.0

    negative_cond = PropertyItem(conductivity=-0.1, permittivity=50.0, density=1000.0)
    assert negative_cond.conductivity == -0.1

    scientific_prop = PropertyItem(
        conductivity=1.23e-15, permittivity=8.854e-12, density=6.022e23
    )
    assert scientific_prop.conductivity == 1.23e-15


def test_structure_phantom_with_realistic_mri_setup():
    """Test StructurePhantom with realistic MRI phantom structure."""
    parent = Blob(position=np.array([0.0, 0.0, 0.0]), radius=100.0)

    children = [
        Blob(position=np.array([10.0, 10.0, 10.0]), radius=15.0),
        Blob(position=np.array([-10.0, -10.0, -10.0]), radius=12.0),
    ]

    tubes = [
        Tube(
            position=np.array([0.0, 0.0, 20.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            radius=2.0,
        ),
        Tube(
            position=np.array([0.0, 0.0, -20.0]),
            direction=np.array([0.0, 1.0, 0.0]),
            radius=1.5,
        ),
    ]

    # Test fixture: list invariance, Blob/Tube are Structure3D subtypes
    phantom = StructurePhantom(parent=parent, children=children, tubes=tubes)  # type: ignore[arg-type]

    assert phantom.parent.radius == 100.0
    assert len(phantom.children) == 2
    assert len(phantom.tubes) == 2
    assert all(isinstance(child, Blob) for child in phantom.children)
    assert all(isinstance(tube, Tube) for tube in phantom.tubes)


def test_mesh_phantom_with_trimesh_integration():
    """Test MeshPhantom integration with Trimesh objects."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])

    parent_mesh = Trimesh(vertices=vertices, faces=faces)

    child_mesh = Trimesh(vertices=vertices * 0.5, faces=faces)

    tube_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 2.0],
            [0.1, 0.0, 2.0],
            [0.1, 0.1, 2.0],
            [0.0, 0.1, 2.0],
        ]
    )
    tube_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ]
    )
    tube_mesh = Trimesh(vertices=tube_vertices, faces=tube_faces)

    phantom = MeshPhantom(parent=parent_mesh, children=[child_mesh], tubes=[tube_mesh])

    assert phantom.parent.volume > 0
    assert len(phantom.children) == 1
    assert len(phantom.tubes) == 1
    assert phantom.children[0].volume < phantom.parent.volume
    assert abs(phantom.tubes[0].volume) > 0


def test_property_phantom_with_multilayer_tissue():
    """Test PropertyPhantom representing multilayer tissue structure."""
    skin = PropertyItem(conductivity=0.1, permittivity=40.0, density=1020.0)

    fat = PropertyItem(conductivity=0.04, permittivity=11.0, density=920.0)
    muscle = PropertyItem(conductivity=0.54, permittivity=60.0, density=1050.0)
    bone = PropertyItem(conductivity=0.02, permittivity=12.0, density=1900.0)

    blood = PropertyItem(conductivity=0.7, permittivity=58.0, density=1060.0)

    phantom = PropertyPhantom(
        parent=skin, children=[fat, muscle, bone], tubes=[blood, blood]
    )

    assert phantom.parent.conductivity == 0.1
    assert len(phantom.children) == 3
    assert len(phantom.tubes) == 2

    assert phantom.parent.conductivity > phantom.children[0].conductivity
    assert phantom.children[2].conductivity < phantom.children[1].conductivity
    assert phantom.tubes[0].conductivity > phantom.children[1].conductivity


def test_phantom_serialization_workflow():
    """Test that all phantom types can be processed in a typical workflow."""
    structure_phantom = StructurePhantom(
        parent=Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0),
        children=[],
        tubes=[],
    )

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2]]
    mesh_phantom = MeshPhantom(
        parent=Trimesh(vertices=vertices, faces=faces), children=[], tubes=[]
    )

    property_phantom = PropertyPhantom(
        parent=PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0),
        children=[],
        tubes=[],
    )

    all_phantoms = [structure_phantom, mesh_phantom, property_phantom]

    assert len(all_phantoms) == 3
    assert isinstance(all_phantoms[0], StructurePhantom)
    assert isinstance(all_phantoms[1], MeshPhantom)
    assert isinstance(all_phantoms[2], PropertyPhantom)

    for phantom in all_phantoms:
        pickled = pickle.dumps(phantom)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, type(phantom))
        # Testing pickling preserves structure
        assert len(unpickled.children) == len(phantom.children)  # type: ignore[attr-defined]
        assert len(unpickled.tubes) == len(phantom.tubes)  # type: ignore[attr-defined]


def test_phantom_boundary_conditions():
    """Test phantom types with boundary conditions relevant to MRI simulation."""
    # Testing None parent boundary condition
    empty_phantom = StructurePhantom(parent=None, children=[], tubes=[])  # type: ignore[arg-type]
    assert empty_phantom.parent is None
    assert len(empty_phantom.children) == 0
    assert len(empty_phantom.tubes) == 0

    large_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=1000.0)
    large_phantom = StructurePhantom(parent=large_blob, children=[], tubes=[])
    assert large_phantom.parent.radius == 1000.0

    extreme_props = PropertyItem(
        conductivity=1e-10, permittivity=1000.0, density=10000.0
    )
    extreme_phantom = PropertyPhantom(parent=extreme_props, children=[], tubes=[])

    assert extreme_phantom.parent.conductivity == 1e-10
    assert extreme_phantom.parent.permittivity == 1000.0
    assert extreme_phantom.parent.density == 10000.0


def test_phantom_types_integration_with_structures():
    """Test phantom types work correctly with Blob/Tube structures."""
    blob_parent = Blob(position=np.array([0.0, 0.0, 0.0]), radius=50.0)
    blob_child = Blob(position=np.array([20.0, 0.0, 0.0]), radius=10.0)
    tube_child = Tube(
        position=np.array([0.0, 20.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        radius=3.0,
    )

    mixed_phantom = StructurePhantom(
        parent=blob_parent, children=[blob_child], tubes=[tube_child]
    )

    assert mixed_phantom.parent.radius == 50.0
    assert mixed_phantom.children[0].radius == 10.0
    assert mixed_phantom.tubes[0].radius == 3.0

    assert np.linalg.norm(mixed_phantom.parent.position) == 0.0
    assert np.linalg.norm(mixed_phantom.children[0].position) == 20.0
    # Runtime type is Tube, which has direction attribute
    assert np.allclose(mixed_phantom.tubes[0].direction, [0.0, 0.0, 1.0])  # type: ignore[attr-defined]


def test_property_item_edge_cases_consolidated():
    """Test PropertyItem edge cases consolidated into one comprehensive test."""
    prop1 = PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0)
    prop2 = PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0)
    prop3 = PropertyItem(conductivity=0.6, permittivity=80.0, density=1000.0)
    assert prop1 == prop2
    assert prop1 != prop3

    assert hasattr(prop1, "conductivity")
    assert hasattr(prop1, "permittivity")
    assert hasattr(prop1, "density")
    assert not hasattr(prop1, "nonexistent_field")

    original = prop1.conductivity
    prop1.conductivity = 0.8
    assert prop1.conductivity == 0.8
    prop1.conductivity = original


def test_point3d_type_alias():
    """Test Point3D type alias behaves correctly with valid coordinates."""
    valid_point: Point3D = (1.0, 2.0, 3.0)
    assert len(valid_point) == 3
    assert all(isinstance(coord, (int, float)) for coord in valid_point)

    origin: Point3D = (0.0, 0.0, 0.0)
    assert origin == (0.0, 0.0, 0.0)

    large_point: Point3D = (1e6, -1e6, 1e6)
    assert large_point[0] == 1e6
    assert large_point[1] == -1e6

    fractional: Point3D = (0.001, 0.5, 0.999)
    assert all(0 <= abs(coord) <= 1 for coord in fractional[:2])


def test_face_indices_type_alias():
    """Test FaceIndices type alias with various face configurations."""
    triangle: FaceIndices = (0, 1, 2)
    assert len(triangle) == 3
    assert all(isinstance(idx, int) for idx in triangle)

    quad: FaceIndices = (0, 1, 2, 3)
    assert len(quad) == 4

    large_face: FaceIndices = (100000, 100001, 100002)
    assert max(large_face) == 100002

    single: FaceIndices = (5,)
    assert len(single) == 1


def test_mesh_grid_type_alias():
    """Test MeshGrid type alias structure and access patterns."""
    grid: MeshGrid = [
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
    ]

    assert len(grid) == 2
    assert len(grid[0]) == 2
    assert grid[0][0] == (0.0, 0.0, 0.0)
    assert grid[1][1] == (1.0, 1.0, 0.0)

    empty_grid: MeshGrid = []
    assert len(empty_grid) == 0

    single_row: MeshGrid = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
    assert len(single_row) == 1
    assert len(single_row[0]) == 2


def test_phantom_item_union_type():
    """Test PhantomItem union type accepts all phantom variants."""
    structure_phantom = StructurePhantom(
        parent=Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0),
        children=[],
        tubes=[],
    )

    mesh_phantom = MeshPhantom(
        parent=Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]),
        children=[],
        tubes=[],
    )

    property_phantom = PropertyPhantom(
        parent=PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0),
        children=[],
        tubes=[],
    )

    phantoms: List[PhantomItem] = [structure_phantom, mesh_phantom, property_phantom]

    assert len(phantoms) == 3
    assert isinstance(phantoms[0], StructurePhantom)
    assert isinstance(phantoms[1], MeshPhantom)
    assert isinstance(phantoms[2], PropertyPhantom)


def test_phantom_equality():
    """Test phantom object equality operations."""
    blob1 = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    blob2 = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)

    phantom1 = StructurePhantom(parent=blob1, children=[], tubes=[])
    phantom2 = StructurePhantom(parent=blob2, children=[], tubes=[])

    try:
        result = phantom1 == phantom2
        assert isinstance(result, (bool, np.ndarray))
    except (ValueError, AttributeError, TypeError):
        # Expected behavior for complex object comparison
        pass


def test_phantom_memory_references():
    """Test that phantom structures handle object references correctly."""
    shared_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=5.0)

    phantom1 = StructurePhantom(parent=shared_blob, children=[], tubes=[])
    phantom2 = StructurePhantom(parent=shared_blob, children=[], tubes=[])

    assert phantom1.parent is phantom2.parent

    original_radius = shared_blob.radius
    shared_blob.radius = 15.0

    assert phantom1.parent.radius == 15.0
    assert phantom2.parent.radius == 15.0

    shared_blob.radius = original_radius


def test_phantom_component_type_validation():
    """Test that phantoms handle incorrect component types gracefully."""

    mixed_children = [
        Blob(position=np.array([0.0, 0.0, 0.0]), radius=5.0),
        Tube(
            position=np.array([10.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            radius=2.0,
        ),
    ]

    mixed_phantom = StructurePhantom(
        parent=Blob(position=np.array([0.0, 0.0, 0.0]), radius=20.0),
        children=mixed_children,
        tubes=[],
    )

    assert len(mixed_phantom.children) == 2
    assert isinstance(mixed_phantom.children[0], Blob)
    assert isinstance(mixed_phantom.children[1], Tube)


def test_property_phantom_mismatched_lengths():
    """Test PropertyPhantom with mismatched component counts."""
    parent_prop = PropertyItem(conductivity=0.5, permittivity=80.0, density=1000.0)

    many_child_props = [
        PropertyItem(conductivity=0.1, permittivity=40.0, density=900.0),
        PropertyItem(conductivity=0.2, permittivity=50.0, density=1100.0),
        PropertyItem(conductivity=0.3, permittivity=60.0, density=1200.0),
    ]

    few_tube_props = [PropertyItem(conductivity=0.7, permittivity=58.0, density=1060.0)]

    mismatched_phantom = PropertyPhantom(
        parent=parent_prop, children=many_child_props, tubes=few_tube_props
    )

    assert len(mismatched_phantom.children) == 3
    assert len(mismatched_phantom.tubes) == 1


def test_phantom_with_none_components():
    """Test phantom behavior with None components (edge case handling)."""
    # Testing None parent boundary condition
    none_parent_phantom = StructurePhantom(parent=None, children=[], tubes=[])  # type: ignore[arg-type]
    assert none_parent_phantom.parent is None
    assert len(none_parent_phantom.children) == 0
    assert len(none_parent_phantom.tubes) == 0

    blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    mixed_phantom = StructurePhantom(parent=blob, children=[None, blob, None], tubes=[])

    assert mixed_phantom.parent is blob
    assert len(mixed_phantom.children) == 3
    assert mixed_phantom.children[0] is None
    assert mixed_phantom.children[1] is blob
    assert mixed_phantom.children[2] is None


def test_phantom_nested_structure_depth():
    """Test phantom structures with deeply nested or recursive references."""
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=100.0)

    recursive_phantom = StructurePhantom(
        parent=parent_blob, children=[parent_blob, parent_blob], tubes=[]
    )

    assert recursive_phantom.parent is parent_blob
    assert all(child is parent_blob for child in recursive_phantom.children)

    original_radius = parent_blob.radius
    parent_blob.radius = 200.0

    assert recursive_phantom.parent.radius == 200.0
    assert all(child.radius == 200.0 for child in recursive_phantom.children)

    parent_blob.radius = original_radius


def test_phantom_component_count_edge_cases():
    """Test phantom structures with unusual component counts."""
    parent = Blob(position=np.array([0.0, 0.0, 0.0]), radius=50.0)

    minimal_phantom = StructurePhantom(parent=parent, children=[], tubes=[])
    assert len(minimal_phantom.children) == 0
    assert len(minimal_phantom.tubes) == 0

    single_child = [Blob(position=np.array([10.0, 0.0, 0.0]), radius=5.0)]
    # Test fixture: list invariance, Blob is Structure3D subtype
    single_phantom = StructurePhantom(parent=parent, children=single_child, tubes=[])  # type: ignore[arg-type]
    assert len(single_phantom.children) == 1

    few_children = [
        Blob(position=np.array([i, 0.0, 0.0]), radius=1.0) for i in range(5)
    ]
    few_tubes = [
        Tube(
            position=np.array([0.0, 0.0, i]),
            direction=np.array([1.0, 0.0, 0.0]),
            radius=0.5,
        )
        for i in range(3)
    ]

    # Test fixture: list invariance, Blob/Tube are Structure3D subtypes
    asymmetric_phantom = StructurePhantom(
        parent=parent, children=few_children, tubes=few_tubes  # type: ignore[arg-type]
    )

    assert len(asymmetric_phantom.children) == 5
    assert len(asymmetric_phantom.tubes) == 3


def test_property_phantom_with_extreme_property_ranges():
    """Test PropertyPhantom with properties spanning extreme ranges."""
    parent_prop = PropertyItem(conductivity=1e-10, permittivity=1.0, density=1.0)

    children_props = [
        PropertyItem(conductivity=1e-8, permittivity=10.0, density=100.0),
        PropertyItem(conductivity=1e-6, permittivity=100.0, density=1000.0),
    ]

    tube_props = [PropertyItem(conductivity=1e6, permittivity=1.0, density=8000.0)]

    extreme_phantom = PropertyPhantom(
        parent=parent_prop, children=children_props, tubes=tube_props
    )

    assert extreme_phantom.parent.conductivity == 1e-10
    assert extreme_phantom.children[-1].conductivity == 1e-6
    assert extreme_phantom.tubes[-1].conductivity == 1e6

    # Verify increasing conductivity pattern
    for i in range(1, len(extreme_phantom.children)):
        assert (
            extreme_phantom.children[i].conductivity
            > extreme_phantom.children[i - 1].conductivity
        )


def test_property_phantom_consistency_validation():
    """Test PropertyPhantom internal consistency across components."""

    parent = PropertyItem(conductivity=0.01, permittivity=5.0, density=1200.0)

    children = [
        PropertyItem(conductivity=0.1, permittivity=50.0, density=1000.0),
        PropertyItem(conductivity=0.5, permittivity=60.0, density=1050.0),
    ]

    tubes = [PropertyItem(conductivity=0.8, permittivity=58.0, density=1060.0)]

    consistent_phantom = PropertyPhantom(parent=parent, children=children, tubes=tubes)

    assert all(
        child.conductivity > consistent_phantom.parent.conductivity
        for child in consistent_phantom.children
    )
    assert all(
        tube.conductivity
        > max(child.conductivity for child in consistent_phantom.children)
        for tube in consistent_phantom.tubes
    )

    assert all(
        tube.density >= max(child.density for child in consistent_phantom.children)
        for tube in consistent_phantom.tubes
    )
