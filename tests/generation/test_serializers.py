import pytest
import numpy as np
from unittest.mock import Mock, patch
from trimesh import Trimesh
import trimesh

from magnet_pinn.generator.serializers import Serializer, MeshSerializer
from magnet_pinn.generator.structures import (
    Structure3D,
    Blob,
    Tube,
    CustomMeshStructure,
)


class ConcreteSerializer(Serializer):
    def serialize(self, structure: Structure3D):
        return "serialized"


class UnsupportedStructure(Structure3D):
    pass


def test_serializer_abstract_base_class_cannot_be_instantiated():
    serializer = Serializer()
    with pytest.raises(NotImplementedError):
        serializer.serialize(None)  # type: ignore[arg-type]  # Testing abstract base class


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


def test_mesh_serializer_serialize_custom_mesh_structure_basic_functionality(tmp_path):
    stl_file = tmp_path / "test_mesh.stl"
    test_mesh = trimesh.primitives.Box(extents=[2.0, 2.0, 2.0])
    test_mesh.export(str(stl_file))

    custom_mesh_structure = CustomMeshStructure(str(stl_file))

    serializer = MeshSerializer()
    mesh = serializer.serialize(custom_mesh_structure)

    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_volume


def test_mesh_serializer_serialize_custom_mesh_structure_returns_copy(tmp_path):
    stl_file = tmp_path / "test_mesh.stl"
    test_mesh = trimesh.primitives.Sphere(radius=1.0)
    test_mesh.export(str(stl_file))

    custom_mesh_structure = CustomMeshStructure(str(stl_file))

    serializer = MeshSerializer()
    mesh1 = serializer.serialize(custom_mesh_structure)
    mesh2 = serializer.serialize(custom_mesh_structure)

    assert mesh1 is not mesh2
    assert mesh1 is not custom_mesh_structure.mesh
    assert mesh2 is not custom_mesh_structure.mesh

    mesh1.apply_scale(2.0)
    assert not np.allclose(mesh1.vertices, mesh2.vertices)


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

    with patch("trimesh.primitives.Sphere") as mock_sphere_class:
        mock_sphere = Mock()
        mock_vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mock_sphere.vertices = mock_vertices
        mock_sphere.faces = [[0, 1, 2]]
        mock_sphere_class.return_value = mock_sphere

        with patch.object(blob, "calculate_offsets") as mock_offsets:
            mock_offsets.return_value = np.array([0.1, 0.2, 0.3])

            with patch("trimesh.Trimesh") as mock_trimesh:
                mock_mesh = Mock()
                mock_trimesh.return_value = mock_mesh

                serializer._serialize_blob(blob, subdivisions=3)

                expected_vertices = (1 + np.array([0.1, 0.2, 0.3])) * mock_vertices
                mock_trimesh.assert_called_once()
                call_args = mock_trimesh.call_args
                np.testing.assert_array_almost_equal(
                    call_args[1]["vertices"], expected_vertices
                )


def test_mesh_serializer_blob_subdivision_parameter_passing():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)

    serializer = MeshSerializer()

    with patch("trimesh.primitives.Sphere") as mock_sphere_class:
        mock_sphere = Mock()
        mock_sphere.vertices = np.array([[1.0, 0.0, 0.0]])
        mock_sphere.faces = [[0, 1, 2]]
        mock_sphere_class.return_value = mock_sphere

        with patch.object(blob, "calculate_offsets"):
            with patch("trimesh.Trimesh"):
                serializer.serialize(blob, subdivisions=7)
                mock_sphere_class.assert_called_once_with(1, subdivisions=7)


def test_mesh_serializer_blob_apply_scale_and_translation():
    position = np.array([2.0, 1.0, -1.0])
    radius = 1.5
    blob = Blob(position=position, radius=radius)

    serializer = MeshSerializer()

    with patch("trimesh.Trimesh") as mock_trimesh:
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

    with patch("trimesh.primitives.Sphere") as mock_sphere_class:
        mock_sphere = Mock()
        test_vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mock_sphere.vertices = test_vertices
        mock_sphere.faces = [[0, 1, 2]]
        mock_sphere_class.return_value = mock_sphere

        with patch("trimesh.Trimesh") as mock_trimesh:
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

    with patch("trimesh.creation.cylinder") as mock_cylinder:
        mock_cylinder.return_value = Mock()

        with patch("trimesh.transformations.translation_matrix") as mock_translation:
            with patch("trimesh.geometry.align_vectors") as mock_align:
                mock_translation.return_value = np.eye(4)
                mock_align.return_value = np.eye(4)

                serializer._serialize_tube(tube, subdivisions=3)

                expected_height = height
                call_args = mock_cylinder.call_args
                assert call_args[1]["radius"] == radius
                assert call_args[1]["height"] == expected_height

                expected_sections = 3**2
                assert call_args[1]["sections"] == expected_sections


def test_mesh_serializer_tube_transformation_composition():
    position = np.array([1.0, 2.0, 3.0])
    direction = np.array([1.0, 1.0, 0.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)

    serializer = MeshSerializer()

    with patch("trimesh.creation.cylinder") as mock_cylinder:
        mock_cylinder.return_value = Mock()

        with patch("trimesh.transformations.translation_matrix") as mock_translation:
            with patch("trimesh.geometry.align_vectors") as mock_align:
                translation_matrix = np.array(
                    [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
                )
                rotation_matrix = np.array(
                    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )

                mock_translation.return_value = translation_matrix
                mock_align.return_value = rotation_matrix

                serializer._serialize_tube(tube, subdivisions=3)

                expected_transform = translation_matrix @ rotation_matrix
                call_args = mock_cylinder.call_args
                np.testing.assert_array_equal(
                    call_args[1]["transform"], expected_transform
                )


def test_mesh_serializer_tube_subdivision_parameter_passing():
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0
    tube = Tube(position=position, direction=direction, radius=radius)

    serializer = MeshSerializer()

    with patch("trimesh.creation.cylinder") as mock_cylinder:
        mock_cylinder.return_value = Mock()

        with patch("trimesh.transformations.translation_matrix"):
            with patch("trimesh.geometry.align_vectors"):
                serializer.serialize(tube, subdivisions=6)

                call_args = mock_cylinder.call_args
                expected_sections = 6**2
                assert call_args[1]["sections"] == expected_sections


def test_mesh_serializer_handles_blob_calculate_offsets_failure():
    position = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    blob = Blob(position=position, radius=radius)

    serializer = MeshSerializer()

    with patch.object(blob, "calculate_offsets") as mock_offsets:
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
    tube = Tube(
        position=position, direction=direction, radius=radius
    )  # No height specified, should default to 10000

    serializer = MeshSerializer()

    with patch("trimesh.creation.cylinder") as mock_cylinder:
        mock_cylinder.return_value = Mock()

        with patch("trimesh.transformations.translation_matrix"):
            with patch("trimesh.geometry.align_vectors"):
                serializer._serialize_tube(tube, subdivisions=3)

                call_args = mock_cylinder.call_args
                expected_height = 10000
                assert call_args[1]["height"] == expected_height


def test_integration_tube_sampler_to_serializer_height_workflow():
    """
    Integration test verifying that TubeSampler creates tubes with
    correct height and MeshSerializer uses it directly.
    """
    from magnet_pinn.generator.samplers import TubeSampler
    from numpy.random import default_rng

    parent_radius = 200.0
    sampler = TubeSampler(
        tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
    )
    serializer = MeshSerializer()

    center = np.array([0.0, 0.0, 0.0])
    ball_radius = 5.0
    tube_radius = 0.5
    rng = default_rng(42)

    tube = sampler._sample_line(center, ball_radius, tube_radius, rng)

    expected_height = 4 * parent_radius
    assert tube.height == expected_height

    mesh = serializer.serialize(tube)

    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_integration_mesh_tube_sampler_to_serializer_height_workflow(tmp_path):
    """
    Integration test verifying that MeshTubeSampler creates tubes with
    correct height and MeshSerializer uses it directly.
    """
    from magnet_pinn.generator.samplers import MeshTubeSampler
    from magnet_pinn.generator.structures import CustomMeshStructure
    from numpy.random import default_rng
    import trimesh

    parent_radius = 150.0
    sampler = MeshTubeSampler(
        tube_max_radius=0.5, tube_min_radius=0.1, parent_radius=parent_radius
    )
    serializer = MeshSerializer()

    stl_file = tmp_path / "test_mesh.stl"
    test_mesh = trimesh.primitives.Box(extents=[2.0, 2.0, 2.0])
    test_mesh.export(str(stl_file))

    parent_mesh_structure = CustomMeshStructure(str(stl_file))
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 2, rng)

    expected_height = 4 * parent_radius
    for tube in tubes:
        assert tube.height == expected_height

        mesh = serializer.serialize(tube)

        assert isinstance(mesh, Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_integration_tube_height_consistency_across_samplers():
    """Test that both TubeSampler and MeshTubeSampler produce tubes with identical height calculation."""
    from magnet_pinn.generator.samplers import TubeSampler, MeshTubeSampler
    from numpy.random import default_rng

    parent_radius_values = [50.0, 100.0, 500.0]

    for parent_radius in parent_radius_values:
        tube_sampler = TubeSampler(
            tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
        )
        mesh_tube_sampler = MeshTubeSampler(
            tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
        )

        rng1 = default_rng(42)
        center = np.array([0.0, 0.0, 0.0])
        ball_radius = 5.0
        tube_radius = 0.5

        tube_from_sampler = tube_sampler._sample_line(
            center, ball_radius, tube_radius, rng1
        )

        expected_height = 4 * parent_radius
        assert tube_from_sampler.height == expected_height
        assert tube_sampler.parent_radius == mesh_tube_sampler.parent_radius


def test_integration_serializer_handles_various_tube_heights():
    """Test that MeshSerializer correctly handles tubes with various heights set by different parent_radius values."""
    serializer = MeshSerializer()

    parent_radius_values = [10.0, 100.0, 1000.0]
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    radius = 1.0

    for parent_radius in parent_radius_values:
        height = 4 * parent_radius
        tube = Tube(
            position=position, direction=direction, radius=radius, height=height
        )

        mesh = serializer.serialize(tube)

        assert isinstance(mesh, Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_integration_default_parent_radius_behavior():
    """Test that default parent_radius behavior is consistent across all components."""
    from magnet_pinn.generator.samplers import TubeSampler, MeshTubeSampler
    from numpy.random import default_rng

    default_parent_radius = 250.0

    tube_sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    mesh_tube_sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    serializer = MeshSerializer()

    assert tube_sampler.parent_radius == default_parent_radius
    assert mesh_tube_sampler.parent_radius == default_parent_radius

    rng = default_rng(42)
    center = np.array([0.0, 0.0, 0.0])
    ball_radius = 5.0
    tube_radius = 0.5

    tube = tube_sampler._sample_line(center, ball_radius, tube_radius, rng)

    expected_height = 4 * default_parent_radius
    assert tube.height == expected_height

    mesh = serializer.serialize(tube)

    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
