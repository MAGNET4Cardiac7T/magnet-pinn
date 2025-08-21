import pytest
import numpy as np
import trimesh
from unittest.mock import Mock, patch, MagicMock

from magnet_pinn.generator.transforms import (
    Transform, Compose, ToMesh, MeshesCleaning, 
    _validate_mesh, _validate_input_meshes
)
from magnet_pinn.generator.typing import StructurePhantom, MeshPhantom
from magnet_pinn.generator.structures import Blob, Tube


class MockTransform(Transform):
    def __call__(self, tissue, *args, **kwargs):
        return tissue


@pytest.fixture
def simple_blob():
    position = np.array([0.0, 0.0, 0.0])
    return Blob(position=position, radius=2.0, seed=42)


@pytest.fixture 
def simple_tube():
    position = np.array([1.0, 1.0, 1.0])
    direction = np.array([0.0, 0.0, 1.0])
    return Tube(position=position, direction=direction, radius=1.0)


@pytest.fixture
def structure_phantom(simple_blob, simple_tube):
    child_blob = Blob(position=np.array([0.5, 0.5, 0.5]), radius=0.8, seed=24)
    return StructurePhantom(
        parent=simple_blob,
        children=[child_blob],
        tubes=[simple_tube]
    )


@pytest.fixture
def simple_sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=1, radius=1.0)


@pytest.fixture
def complex_mesh_phantom(simple_sphere_mesh):
    parent_mesh = simple_sphere_mesh.copy()
    child_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    tube_mesh = trimesh.creation.cylinder(radius=0.3, height=2.0)
    
    return MeshPhantom(
        parent=parent_mesh,
        children=[child_mesh],
        tubes=[tube_mesh]
    )


def test_abstract_transform_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Transform()


def test_mock_transform_repr_returns_class_name():
    transform = MockTransform()
    assert repr(transform) == "MockTransform()"


def test_compose_initialization_with_single_transform():
    transform = MockTransform()
    compose = Compose([transform])
    assert len(compose.transforms) == 1
    assert compose.transforms[0] is transform


def test_compose_initialization_with_multiple_transforms():
    transforms = [MockTransform(), MockTransform(), MockTransform()]
    compose = Compose(transforms)
    assert len(compose.transforms) == 3
    assert compose.transforms == transforms


def test_compose_initialization_with_empty_list():
    compose = Compose([])
    assert len(compose.transforms) == 0


def test_compose_call_applies_transforms_sequentially(structure_phantom):
    mock1 = Mock(spec=Transform)
    mock2 = Mock(spec=Transform)
    mock3 = Mock(spec=Transform)
    
    mock1.return_value = "result1"
    mock2.return_value = "result2" 
    mock3.return_value = "result3"
    
    compose = Compose([mock1, mock2, mock3])
    result = compose(structure_phantom)
    
    mock1.assert_called_once_with(structure_phantom, structure_phantom)
    mock2.assert_called_once_with("result1", structure_phantom)
    mock3.assert_called_once_with("result2", structure_phantom)
    assert result == "result3"


def test_compose_call_with_no_transforms_returns_input(structure_phantom):
    compose = Compose([])
    result = compose(structure_phantom)
    assert result is structure_phantom


def test_compose_call_passes_additional_arguments(structure_phantom):
    mock_transform = Mock(spec=Transform)
    mock_transform.return_value = structure_phantom
    
    compose = Compose([mock_transform])
    compose(structure_phantom, "extra_arg", keyword="extra_kwarg")
    
    mock_transform.assert_called_once_with(structure_phantom, structure_phantom, "extra_arg", keyword="extra_kwarg")


def test_compose_repr_shows_component_transforms():
    transforms = [MockTransform(), MockTransform()]
    compose = Compose(transforms)
    expected = "Compose(MockTransform(), MockTransform())"
    assert repr(compose) == expected


def test_tomesh_initialization_creates_serializer():
    tomesh = ToMesh()
    assert hasattr(tomesh, 'serializer')
    assert tomesh.serializer is not None


def test_tomesh_call_returns_mesh_phantom(structure_phantom):
    tomesh = ToMesh()
    
    with patch.object(tomesh.serializer, 'serialize') as mock_serialize:
        mock_mesh = Mock(spec=trimesh.Trimesh)
        mock_serialize.return_value = mock_mesh
        
        result = tomesh(structure_phantom)
        
        assert isinstance(result, MeshPhantom)
        assert result.parent is mock_mesh
        assert len(result.children) == 1
        assert result.children[0] is mock_mesh
        assert len(result.tubes) == 1
        assert result.tubes[0] is mock_mesh


def test_tomesh_call_serializes_all_components(structure_phantom):
    tomesh = ToMesh()
    
    with patch.object(tomesh.serializer, 'serialize') as mock_serialize:
        mock_serialize.return_value = Mock(spec=trimesh.Trimesh)
        
        tomesh(structure_phantom)
        
        assert mock_serialize.call_count == 3
        mock_serialize.assert_any_call(structure_phantom.parent)
        mock_serialize.assert_any_call(structure_phantom.children[0])
        mock_serialize.assert_any_call(structure_phantom.tubes[0])


def test_validate_mesh_accepts_valid_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    _validate_mesh(mesh, "test_operation")


def test_validate_mesh_rejects_none_mesh():
    with pytest.raises(ValueError, match="Mesh is None after test_operation"):
        _validate_mesh(None, "test_operation")


def test_validate_mesh_rejects_mesh_with_no_vertices():
    mesh = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
    with pytest.raises(ValueError, match="Mesh has no vertices after test_operation"):
        _validate_mesh(mesh, "test_operation")


def test_validate_mesh_rejects_mesh_with_no_faces():
    mesh = trimesh.Trimesh(vertices=np.array([[0, 0, 0]]), faces=np.array([]))
    with pytest.raises(ValueError, match="Mesh has no faces after test_operation"):
        _validate_mesh(mesh, "test_operation")


def test_validate_mesh_rejects_mesh_with_zero_volume():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    with pytest.raises(ValueError, match="Mesh has invalid volume"):
        _validate_mesh(mesh, "test_operation")


def test_validate_input_meshes_rejects_none_in_list():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    with pytest.raises(ValueError, match="Input mesh 1 is None for test_operation"):
        _validate_input_meshes([mesh, None], "test_operation")


def test_validate_input_meshes_rejects_empty_vertices():
    mesh1 = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh2 = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
    with pytest.raises(ValueError, match="Input mesh 1 has no vertices for test_operation"):
        _validate_input_meshes([mesh1, mesh2], "test_operation")


def test_validate_input_meshes_rejects_empty_faces():
    mesh1 = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh2 = trimesh.Trimesh(vertices=np.array([[0, 0, 0]]), faces=np.array([]))
    with pytest.raises(ValueError, match="Input mesh 1 has no faces for test_operation"):
        _validate_input_meshes([mesh1, mesh2], "test_operation")


def test_meshes_cleaning_call_returns_mesh_phantom(complex_mesh_phantom):
    cleaning = MeshesCleaning()
    
    with patch.object(cleaning, '_clean_mesh') as mock_clean:
        mock_clean.return_value = Mock(spec=trimesh.Trimesh)
        
        result = cleaning(complex_mesh_phantom)
        
        assert isinstance(result, MeshPhantom)
        expected_calls = 1 + len(complex_mesh_phantom.children) + len(complex_mesh_phantom.tubes)
        assert mock_clean.call_count == expected_calls


def test_meshes_cleaning_clean_mesh_applies_correct_sequence():
    cleaning = MeshesCleaning()
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    original_vertex_count = len(mesh.vertices)
    original_face_count = len(mesh.faces)
    
    result = cleaning._clean_mesh(mesh)
    
    assert isinstance(result, trimesh.Trimesh)
    assert result is not mesh
    assert len(result.vertices) > 0
    assert len(result.faces) > 0


def test_meshes_cleaning_clean_mesh_preserves_mesh_type():
    cleaning = MeshesCleaning()
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    result = cleaning._clean_mesh(mesh)
    
    assert isinstance(result, trimesh.Trimesh)


def test_meshes_cleaning_handles_empty_children_and_tubes():
    cleaning = MeshesCleaning()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    phantom = MeshPhantom(parent=parent_mesh, children=[], tubes=[])
    
    result = cleaning(phantom)
    
    assert isinstance(result, MeshPhantom)
    assert len(result.children) == 0
    assert len(result.tubes) == 0


def test_transforms_work_with_real_meshes():
    parent_sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    child_sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.8)
    child_sphere.vertices += [0.5, 0.5, 0.5]
    tube_cylinder = trimesh.creation.cylinder(radius=0.3, height=1.0)
    
    phantom = MeshPhantom(
        parent=parent_sphere,
        children=[child_sphere],
        tubes=[tube_cylinder]
    )
    
    cleaning = MeshesCleaning()
    result = cleaning(phantom)
    
    assert isinstance(result, MeshPhantom)
    assert result.parent.is_volume
    assert len(result.children) == 1
    assert result.children[0].is_volume
    assert len(result.tubes) == 1
    assert result.tubes[0].is_volume


def test_abstract_transform_call_raises_not_implemented():
    class IncompleteTransform(Transform):
        pass
    
    with pytest.raises(TypeError):
        IncompleteTransform()


def test_abstract_transform_call_method_raises_not_implemented():
    class IncompleteTransform(Transform):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)
    
    transform = IncompleteTransform()
    with pytest.raises(NotImplementedError, match="Subclasses must implement `__call__` method"):
        transform("dummy_input")


def test_compose_initialization_validates_transform_interface():
    transform = MockTransform()
    compose = Compose([transform])
    assert all(hasattr(t, '__call__') for t in compose.transforms)


def test_tomesh_call_preserves_phantom_structure_integrity(structure_phantom):
    tomesh = ToMesh()
    
    with patch.object(tomesh.serializer, 'serialize') as mock_serialize:
        mock_mesh = Mock(spec=trimesh.Trimesh)
        mock_serialize.return_value = mock_mesh
        
        result = tomesh(structure_phantom)
        
        assert len(result.children) == len(structure_phantom.children)
        assert len(result.tubes) == len(structure_phantom.tubes)


def test_meshes_cleaning_preserves_phantom_component_counts(complex_mesh_phantom):
    cleaning = MeshesCleaning()
    
    result = cleaning(complex_mesh_phantom)
    
    assert len(result.children) == len(complex_mesh_phantom.children)
    assert len(result.tubes) == len(complex_mesh_phantom.tubes)
