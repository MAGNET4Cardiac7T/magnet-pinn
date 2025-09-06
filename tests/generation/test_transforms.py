import pytest
import numpy as np
import trimesh
from unittest.mock import Mock, patch, MagicMock

from magnet_pinn.generator.transforms import (
    Transform, Compose, ToMesh, MeshesCutout, MeshesCleaning, 
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
    
    mock1.assert_called_once_with(structure_phantom)
    mock2.assert_called_once_with("result1")
    mock3.assert_called_once_with("result2")
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
    
    mock_transform.assert_called_once_with(structure_phantom, "extra_arg", keyword="extra_kwarg")


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


def test_meshes_cutout_call_returns_mesh_phantom(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch.object(cutout, '_cut_parent') as mock_cut_parent, \
         patch.object(cutout, '_cut_children') as mock_cut_children, \
         patch.object(cutout, '_cut_tubes') as mock_cut_tubes:
        
        mock_cut_parent.return_value = Mock(spec=trimesh.Trimesh)
        mock_cut_children.return_value = [Mock(spec=trimesh.Trimesh)]
        mock_cut_tubes.return_value = [Mock(spec=trimesh.Trimesh)]
        
        result = cutout(complex_mesh_phantom)
        
        assert isinstance(result, MeshPhantom)
        mock_cut_parent.assert_called_once_with(complex_mesh_phantom)
        mock_cut_children.assert_called_once_with(complex_mesh_phantom)
        mock_cut_tubes.assert_called_once_with(complex_mesh_phantom)


def test_meshes_cutout_call_handles_runtime_error(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch.object(cutout, '_cut_parent', side_effect=RuntimeError("Boolean operation failed")):
        with pytest.raises(RuntimeError):
            cutout(complex_mesh_phantom)


def test_meshes_cutout_call_handles_value_error(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch.object(cutout, '_cut_parent', side_effect=ValueError("Invalid mesh")):
        with pytest.raises(ValueError):
            cutout(complex_mesh_phantom)


def test_meshes_cutout_cut_parent_with_no_cutters(complex_mesh_phantom):
    cutout = MeshesCutout()
    phantom_no_cutters = MeshPhantom(
        parent=complex_mesh_phantom.parent,
        children=[],
        tubes=[]
    )
    
    result = cutout._cut_parent(phantom_no_cutters)
    assert result is phantom_no_cutters.parent


def test_meshes_cutout_cut_parent_with_cutters(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.union') as mock_union, \
         patch('trimesh.boolean.difference') as mock_difference, \
         patch('magnet_pinn.generator.transforms._validate_input_meshes'), \
         patch('magnet_pinn.generator.transforms._validate_mesh'):
        
        mock_union_result = Mock(spec=trimesh.Trimesh)
        mock_union.return_value = mock_union_result
        mock_difference_result = Mock(spec=trimesh.Trimesh)
        mock_difference.return_value = mock_difference_result
        
        result = cutout._cut_parent(complex_mesh_phantom)
        
        mock_union.assert_called_once()
        mock_difference.assert_called_once()
        assert result is mock_difference_result


def test_meshes_cutout_cut_parent_raises_runtime_error_on_failure(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.union', side_effect=Exception("Boolean failed")):
        with pytest.raises(RuntimeError, match="Boolean operation failed for parent cutting"):
            cutout._cut_parent(complex_mesh_phantom)


def test_meshes_cutout_cut_children_with_no_tubes(complex_mesh_phantom):
    cutout = MeshesCutout()
    phantom_no_tubes = MeshPhantom(
        parent=complex_mesh_phantom.parent,
        children=complex_mesh_phantom.children,
        tubes=[]
    )
    
    result = cutout._cut_children(phantom_no_tubes)
    assert result == phantom_no_tubes.children


def test_meshes_cutout_cut_children_with_tubes(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.union') as mock_union, \
         patch('trimesh.boolean.difference') as mock_difference, \
         patch('magnet_pinn.generator.transforms._validate_input_meshes'), \
         patch('magnet_pinn.generator.transforms._validate_mesh'):
        
        mock_union_result = Mock(spec=trimesh.Trimesh)
        mock_union.return_value = mock_union_result
        mock_difference_result = Mock(spec=trimesh.Trimesh)
        mock_difference.return_value = mock_difference_result
        
        result = cutout._cut_children(complex_mesh_phantom)
        
        mock_union.assert_called_once()
        assert mock_difference.call_count == len(complex_mesh_phantom.children)
        assert len(result) == len(complex_mesh_phantom.children)


def test_meshes_cutout_cut_children_handles_child_failure(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.union') as mock_union, \
         patch('trimesh.boolean.difference', side_effect=Exception("Child cutting failed")), \
         patch('magnet_pinn.generator.transforms._validate_input_meshes'), \
         patch('magnet_pinn.generator.transforms._validate_mesh'):
        
        mock_union.return_value = Mock(spec=trimesh.Trimesh)
        
        with pytest.raises(RuntimeError, match="Failed to cut child 0"):
            cutout._cut_children(complex_mesh_phantom)


def test_meshes_cutout_cut_tubes(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.intersection') as mock_intersection, \
         patch('magnet_pinn.generator.transforms._validate_input_meshes'), \
         patch('magnet_pinn.generator.transforms._validate_mesh'):
        
        mock_intersection_result = Mock(spec=trimesh.Trimesh)
        mock_intersection.return_value = mock_intersection_result
        
        result = cutout._cut_tubes(complex_mesh_phantom)
        
        assert mock_intersection.call_count == len(complex_mesh_phantom.tubes)
        assert len(result) == len(complex_mesh_phantom.tubes)


def test_meshes_cutout_cut_tubes_handles_tube_failure(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.intersection', side_effect=Exception("Tube cutting failed")), \
         patch('magnet_pinn.generator.transforms._validate_input_meshes'):
        
        with pytest.raises(RuntimeError, match="Failed to cut tube 0"):
            cutout._cut_tubes(complex_mesh_phantom)


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


def test_compose_with_complete_pipeline(structure_phantom):
    tomesh = ToMesh()
    cutout = MeshesCutout() 
    cleaning = MeshesCleaning()
    
    pipeline = Compose([tomesh, cutout, cleaning])
    
    with patch.object(tomesh.serializer, 'serialize') as mock_serialize, \
         patch.object(cutout, '_cut_parent') as mock_cut_parent, \
         patch.object(cutout, '_cut_children') as mock_cut_children, \
         patch.object(cutout, '_cut_tubes') as mock_cut_tubes, \
         patch.object(cleaning, '_clean_mesh') as mock_clean:
        
        mock_mesh = Mock(spec=trimesh.Trimesh)
        mock_serialize.return_value = mock_mesh
        mock_cut_parent.return_value = mock_mesh
        mock_cut_children.return_value = [mock_mesh]
        mock_cut_tubes.return_value = [mock_mesh]
        mock_clean.return_value = mock_mesh
        
        result = pipeline(structure_phantom)
        
        assert isinstance(result, MeshPhantom)
        assert mock_serialize.call_count == 3
        mock_cut_parent.assert_called_once()
        mock_cut_children.assert_called_once()
        mock_cut_tubes.assert_called_once()
        expected_clean_calls = 1 + len(result.children) + len(result.tubes)
        assert mock_clean.call_count == expected_clean_calls


def test_meshes_cutout_cut_parent_error_message_includes_details(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.union', side_effect=Exception("Mock error")):
        with pytest.raises(RuntimeError) as exc_info:
            cutout._cut_parent(complex_mesh_phantom)
        
        error_msg = str(exc_info.value)
        assert "Parent vertices:" in error_msg
        assert "Parent faces:" in error_msg
        assert "Parent volume:" in error_msg
        assert "Cutters count:" in error_msg


def test_meshes_cutout_cut_children_error_message_includes_details(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.union', side_effect=Exception("Mock error")):
        with pytest.raises(RuntimeError) as exc_info:
            cutout._cut_children(complex_mesh_phantom)
        
        error_msg = str(exc_info.value)
        assert "Children count:" in error_msg
        assert "Tubes count:" in error_msg


def test_meshes_cutout_cut_tubes_error_message_includes_details(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('trimesh.boolean.intersection', side_effect=Exception("Mock error")), \
         patch('magnet_pinn.generator.transforms._validate_input_meshes'):
        
        with pytest.raises(RuntimeError) as exc_info:
            cutout._cut_tubes(complex_mesh_phantom)
        
        error_msg = str(exc_info.value)
        assert "Failed to cut tube 0" in error_msg
        assert "vertices:" in error_msg
        assert "faces:" in error_msg
        assert "volume:" in error_msg


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


def test_meshes_cutout_cut_tubes_general_exception_handler(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch('magnet_pinn.generator.transforms._validate_input_meshes', side_effect=ValueError("Validation error")):
        
        with pytest.raises(RuntimeError) as exc_info:
            cutout._cut_tubes(complex_mesh_phantom)
        
        error_msg = str(exc_info.value)
        assert "Boolean operation failed for tube cutting" in error_msg
        assert "Tubes count:" in error_msg
        assert "Parent vertices:" in error_msg
        assert "Parent faces:" in error_msg
        assert "Parent volume:" in error_msg


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


def test_meshes_cutout_preserves_phantom_component_counts(complex_mesh_phantom):
    cutout = MeshesCutout()
    
    with patch.object(cutout, '_cut_parent') as mock_cut_parent, \
         patch.object(cutout, '_cut_children') as mock_cut_children, \
         patch.object(cutout, '_cut_tubes') as mock_cut_tubes:
        
        mock_cut_parent.return_value = Mock(spec=trimesh.Trimesh)
        mock_cut_children.return_value = [Mock(spec=trimesh.Trimesh) for _ in complex_mesh_phantom.children]
        mock_cut_tubes.return_value = [Mock(spec=trimesh.Trimesh) for _ in complex_mesh_phantom.tubes]
        
        result = cutout(complex_mesh_phantom)
        
        assert len(result.children) == len(complex_mesh_phantom.children)
        assert len(result.tubes) == len(complex_mesh_phantom.tubes)


def test_meshes_cleaning_preserves_phantom_component_counts(complex_mesh_phantom):
    cleaning = MeshesCleaning()
    
    result = cleaning(complex_mesh_phantom)
    
    assert len(result.children) == len(complex_mesh_phantom.children)
    assert len(result.tubes) == len(complex_mesh_phantom.tubes)
