import pytest
import numpy as np
from unittest.mock import Mock, patch
from trimesh import Trimesh

from magnet_pinn.generator.serializers import Serializer, MeshSerializer
from magnet_pinn.generator.structures import Structure3D, Blob, Tube


class ConcreteSerializer(Serializer):
    def serialize(self, structure: Structure3D):
        return "serialized"


class UnsupportedStructure(Structure3D):
    pass


def test_serializer_abstract_base_class_cannot_be_instantiated():
    serializer = Serializer()
    with pytest.raises(NotImplementedError):
        serializer.serialize(None)


def test_mesh_serializer_initialization():
    serializer = MeshSerializer()
    assert isinstance(serializer, Serializer)
    assert isinstance(serializer, MeshSerializer)


def test_mesh_serializer_serialize_unsupported_structure_raises_value_error():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    unsupported_structure = UnsupportedStructure(position=position, radius=radius)
    
    serializer = MeshSerializer()
    with pytest.raises(ValueError, match="Unsupported structure type"):
        serializer.serialize(unsupported_structure)


def test_mesh_serializer_serialize_blob_basic_functionality():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    mesh = serializer.serialize(blob)
    
    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_volume


def test_mesh_serializer_serialize_tube_basic_functionality():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)
    
    serializer = MeshSerializer()
    mesh = serializer.serialize(tube)
    
    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_mesh_serializer_blob_vertex_transformation_algorithm():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.primitives.Sphere') as mock_sphere_class:
        mock_sphere = Mock()
        mock_vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mock_sphere.vertices = mock_vertices
        mock_sphere.faces = [[0, 1, 2]]
        mock_sphere_class.return_value = mock_sphere
        
        with patch.object(blob, 'calculate_offsets') as mock_offsets:
            mock_offsets.return_value = np.array([0.1, 0.2, 0.3])
            
            with patch('trimesh.Trimesh') as mock_trimesh:
                mock_mesh = Mock()
                mock_trimesh.return_value = mock_mesh
                
                serializer._serialize_blob(blob, subdivisions=3)
                
                expected_vertices = (1 + np.array([0.1, 0.2, 0.3])) * mock_vertices
                mock_trimesh.assert_called_once()
                call_args = mock_trimesh.call_args
                np.testing.assert_array_almost_equal(call_args[1]['vertices'], expected_vertices)


def test_mesh_serializer_blob_subdivision_parameter_passing():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.primitives.Sphere') as mock_sphere_class:
        mock_sphere = Mock()
        mock_sphere.vertices = np.array([[1.0, 0.0, 0.0]])
        mock_sphere.faces = [[0, 1, 2]]
        mock_sphere_class.return_value = mock_sphere
        
        with patch.object(blob, 'calculate_offsets'):
            with patch('trimesh.Trimesh'):
                serializer.serialize(blob, subdivisions=7)
                mock_sphere_class.assert_called_once_with(1, subdivisions=7)


def test_mesh_serializer_blob_apply_scale_and_translation():
    position = np.array([2.0, 1.0, -1.0])
    radius = 1.5
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.Trimesh') as mock_trimesh:
        mock_mesh = Mock()
        mock_trimesh.return_value = mock_mesh
        
        serializer._serialize_blob(blob, subdivisions=3)
        
        mock_mesh.apply_scale.assert_called_once_with(radius)
        mock_mesh.apply_translation.assert_called_once()
        
        translation_call = mock_mesh.apply_translation.call_args[0][0]
        np.testing.assert_array_equal(translation_call, position)


def test_mesh_serializer_blob_calculate_offsets_integration():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.primitives.Sphere') as mock_sphere_class:
        mock_sphere = Mock()
        test_vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mock_sphere.vertices = test_vertices
        mock_sphere.faces = [[0, 1, 2]]
        mock_sphere_class.return_value = mock_sphere
        
        with patch('trimesh.Trimesh') as mock_trimesh:
            mock_mesh = Mock()
            mock_trimesh.return_value = mock_mesh
            
            serializer._serialize_blob(blob, subdivisions=3)


def test_mesh_serializer_tube_parameter_calculation():
    position = np.array([1.0, 2.0, 3.0])
    direction = np.array([1.0, 1.0, 0.0])
    radius = 2.0
    height = 1.5
    tube = Tube(position=position, direction=direction, radius=radius, height=height)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.creation.cylinder') as mock_cylinder:
        mock_cylinder.return_value = Mock()
        
        with patch('trimesh.transformations.translation_matrix') as mock_translation:
            with patch('trimesh.geometry.align_vectors') as mock_align:
                mock_translation.return_value = np.eye(4)
                mock_align.return_value = np.eye(4)
                
                serializer._serialize_tube(tube, subdivisions=3)
                
                expected_height = height * radius
                call_args = mock_cylinder.call_args
                assert call_args[1]['radius'] == radius
                assert call_args[1]['height'] == expected_height
                
                expected_sections = 3 ** 2
                assert call_args[1]['sections'] == expected_sections


def test_mesh_serializer_tube_transformation_composition():
    position = np.array([1.0, 2.0, 3.0])
    direction = np.array([1.0, 1.0, 0.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.creation.cylinder') as mock_cylinder:
        mock_cylinder.return_value = Mock()
        
        with patch('trimesh.transformations.translation_matrix') as mock_translation:
            with patch('trimesh.geometry.align_vectors') as mock_align:
                translation_matrix = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
                rotation_matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                
                mock_translation.return_value = translation_matrix
                mock_align.return_value = rotation_matrix
                
                serializer._serialize_tube(tube, subdivisions=3)
                
                expected_transform = translation_matrix @ rotation_matrix
                call_args = mock_cylinder.call_args
                np.testing.assert_array_equal(call_args[1]['transform'], expected_transform)


def test_mesh_serializer_tube_subdivision_parameter_passing():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch('trimesh.creation.cylinder') as mock_cylinder:
        mock_cylinder.return_value = Mock()
        
        with patch('trimesh.transformations.translation_matrix'):
            with patch('trimesh.geometry.align_vectors'):
                serializer.serialize(tube, subdivisions=6)
                
                call_args = mock_cylinder.call_args
                expected_sections = 6 ** 2
                assert call_args[1]['sections'] == expected_sections


def test_mesh_serializer_handles_blob_calculate_offsets_failure():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    
    with patch.object(blob, 'calculate_offsets') as mock_offsets:
        mock_offsets.side_effect = RuntimeError("Calculation failed")
        
        with pytest.raises(RuntimeError, match="Calculation failed"):
            serializer._serialize_blob(blob, subdivisions=3)


def test_mesh_serializer_blob_with_default_subdivisions():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    mesh = serializer.serialize(blob)
    
    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_volume


def test_mesh_serializer_tube_with_default_subdivisions():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)
    
    serializer = MeshSerializer()
    mesh = serializer.serialize(tube)
    
    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_volume


def test_mesh_serializer_blob_with_custom_subdivisions():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    mesh_low = serializer.serialize(blob, subdivisions=2)
    mesh_high = serializer.serialize(blob, subdivisions=6)
    
    assert isinstance(mesh_low, Trimesh)
    assert isinstance(mesh_high, Trimesh)
    assert len(mesh_high.vertices) > len(mesh_low.vertices)


def test_mesh_serializer_tube_with_custom_subdivisions():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)
    
    serializer = MeshSerializer()
    mesh_low = serializer.serialize(tube, subdivisions=2)
    mesh_high = serializer.serialize(tube, subdivisions=6)
    
    assert isinstance(mesh_low, Trimesh)
    assert isinstance(mesh_high, Trimesh)
    assert len(mesh_high.faces) > len(mesh_low.faces)


def test_mesh_serializer_private_method_blob():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)
    
    serializer = MeshSerializer()
    mesh = serializer._serialize_blob(blob, subdivisions=3)
    
    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_mesh_serializer_private_method_tube():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)
    
    serializer = MeshSerializer()
    mesh = serializer._serialize_tube(tube, subdivisions=3)
    
    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_mesh_serializer_tube_default_height_parameter():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)  # No height specified, should default to 10000
    
    serializer = MeshSerializer()
    
    with patch('trimesh.creation.cylinder') as mock_cylinder:
        mock_cylinder.return_value = Mock()
        
        with patch('trimesh.transformations.translation_matrix'):
            with patch('trimesh.geometry.align_vectors'):
                serializer._serialize_tube(tube, subdivisions=3)
                
                call_args = mock_cylinder.call_args
                expected_height = 10000 * radius  # Default height * radius
                assert call_args[1]['height'] == expected_height
