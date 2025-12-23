import pytest
import numpy as np
import trimesh
from unittest.mock import Mock, patch

from magnet_pinn.generator.transforms import (
    Transform,
    Compose,
    ToMesh,
    MeshesCleaning,
    MeshesRemesh,
    MeshesTubesClipping,
    MeshesChildrenCutout,
    MeshesParentCutoutWithChildren,
    MeshesParentCutoutWithTubes,
    MeshesChildrenClipping,
    _validate_mesh,
    _validate_input_meshes,
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
        parent=simple_blob, children=[child_blob], tubes=[simple_tube]
    )


@pytest.fixture
def simple_sphere_mesh():
    return trimesh.creation.icosphere(subdivisions=1, radius=1.0)


@pytest.fixture
def complex_mesh_phantom(simple_sphere_mesh):
    parent_mesh = simple_sphere_mesh.copy()
    child_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    tube_mesh = trimesh.creation.cylinder(radius=0.3, height=2.0)

    return MeshPhantom(parent=parent_mesh, children=[child_mesh], tubes=[tube_mesh])


def test_abstract_transform_cannot_be_instantiated():
    # Testing that abstract Transform class cannot be instantiated
    with pytest.raises(TypeError):
        Transform()  # type: ignore[abstract]


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
    # Test fixture: list invariance, MockTransform is Transform subtype
    compose = Compose(transforms)  # type: ignore[arg-type]
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

    mock_transform.assert_called_once_with(
        structure_phantom, structure_phantom, "extra_arg", keyword="extra_kwarg"
    )


def test_compose_repr_shows_component_transforms():
    transforms = [MockTransform(), MockTransform()]
    # Test fixture: list invariance, MockTransform is Transform subtype
    compose = Compose(transforms)  # type: ignore[arg-type]
    expected = "Compose(MockTransform(), MockTransform())"
    assert repr(compose) == expected


def test_tomesh_initialization_creates_serializer():
    tomesh = ToMesh()
    assert hasattr(tomesh, "serializer")
    assert tomesh.serializer is not None


def test_tomesh_call_returns_mesh_phantom(structure_phantom):
    tomesh = ToMesh()

    with patch.object(tomesh.serializer, "serialize") as mock_serialize:
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

    with patch.object(tomesh.serializer, "serialize") as mock_serialize:
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
    # Testing None validation in _validate_mesh
    with pytest.raises(ValueError, match="Mesh is None after test_operation"):
        _validate_mesh(None, "test_operation")  # type: ignore[arg-type]


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
    # Testing None validation in mesh list
    with pytest.raises(ValueError, match="Input mesh 1 is None for test_operation"):
        _validate_input_meshes([mesh, None], "test_operation")  # type: ignore[list-item]


def test_validate_input_meshes_rejects_empty_vertices():
    mesh1 = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh2 = trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
    with pytest.raises(
        ValueError, match="Input mesh 1 has no vertices for test_operation"
    ):
        _validate_input_meshes([mesh1, mesh2], "test_operation")


def test_validate_input_meshes_rejects_empty_faces():
    mesh1 = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh2 = trimesh.Trimesh(vertices=np.array([[0, 0, 0]]), faces=np.array([]))
    with pytest.raises(
        ValueError, match="Input mesh 1 has no faces for test_operation"
    ):
        _validate_input_meshes([mesh1, mesh2], "test_operation")


def test_meshes_cleaning_call_returns_mesh_phantom(complex_mesh_phantom):
    cleaning = MeshesCleaning()

    with patch.object(cleaning, "_clean_mesh") as mock_clean:
        mock_clean.return_value = Mock(spec=trimesh.Trimesh)

        result = cleaning(complex_mesh_phantom)

        assert isinstance(result, MeshPhantom)
        expected_calls = (
            1 + len(complex_mesh_phantom.children) + len(complex_mesh_phantom.tubes)
        )
        assert mock_clean.call_count == expected_calls


def test_meshes_cleaning_clean_mesh_applies_correct_sequence():
    cleaning = MeshesCleaning()
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

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
        parent=parent_sphere, children=[child_sphere], tubes=[tube_cylinder]
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

    # Testing that incomplete abstract Transform subclass cannot be instantiated
    with pytest.raises(TypeError):
        IncompleteTransform()  # type: ignore[abstract]


def test_abstract_transform_call_method_raises_not_implemented():
    class IncompleteTransform(Transform):
        def __call__(self, *args, **kwargs):
            # Testing abstract method behavior via super() call
            return super().__call__(*args, **kwargs)  # type: ignore[safe-super]

    transform = IncompleteTransform()
    with pytest.raises(
        NotImplementedError, match="Subclasses must implement `__call__` method"
    ):
        transform("dummy_input")


def test_compose_initialization_validates_transform_interface():
    transform = MockTransform()
    compose = Compose([transform])
    assert all(hasattr(t, "__call__") for t in compose.transforms)


def test_tomesh_call_preserves_phantom_structure_integrity(structure_phantom):
    tomesh = ToMesh()

    with patch.object(tomesh.serializer, "serialize") as mock_serialize:
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


def test_meshes_remesh_initialization():
    remesh = MeshesRemesh()
    assert remesh.max_len == 8.0


def test_meshes_remesh_initialization_with_custom_max_len():
    remesh = MeshesRemesh(max_len=5.0)
    assert remesh.max_len == 5.0


def test_meshes_remesh_call_returns_mesh_phantom(complex_mesh_phantom):
    remesh = MeshesRemesh(max_len=10.0)

    with patch("trimesh.remesh.subdivide_to_size") as mock_subdivide:
        mock_subdivide.return_value = (
            complex_mesh_phantom.parent.vertices,
            complex_mesh_phantom.parent.faces,
        )

        result = remesh(complex_mesh_phantom)

        assert isinstance(result, MeshPhantom)
        assert mock_subdivide.call_count == 3


def test_meshes_remesh_preserves_phantom_structure(complex_mesh_phantom):
    remesh = MeshesRemesh(max_len=10.0)

    with patch("trimesh.remesh.subdivide_to_size") as mock_subdivide:
        mock_subdivide.return_value = (
            complex_mesh_phantom.parent.vertices,
            complex_mesh_phantom.parent.faces,
        )

        result = remesh(complex_mesh_phantom)

        assert len(result.children) == len(complex_mesh_phantom.children)
        assert len(result.tubes) == len(complex_mesh_phantom.tubes)


def test_meshes_remesh_calls_subdivide_with_correct_parameters(complex_mesh_phantom):
    remesh = MeshesRemesh(max_len=5.0)

    with patch("trimesh.remesh.subdivide_to_size") as mock_subdivide:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mock_subdivide.return_value = (vertices, faces)

        remesh(complex_mesh_phantom)

        for call in mock_subdivide.call_args_list:
            assert call[1]["max_edge"] == 5.0


def test_meshes_tubes_clipping_successful_operation(complex_mesh_phantom):
    clipping = MeshesTubesClipping()

    mock_clipped_tube = Mock(spec=trimesh.Trimesh)
    mock_clipped_tube.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_clipped_tube.faces = np.array([[0, 1, 2]])
    mock_clipped_tube.volume = 1.0
    mock_clipped_tube.is_volume = True
    mock_clipped_tube.remove_degenerate_faces = Mock()
    mock_clipped_tube.remove_duplicate_faces = Mock()
    mock_clipped_tube.remove_unreferenced_vertices = Mock()
    mock_clipped_tube.fill_holes = Mock()

    with patch(
        "trimesh.boolean.intersection", return_value=mock_clipped_tube
    ) as mock_intersection:
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with patch("magnet_pinn.generator.transforms._validate_mesh"):
                result = clipping(complex_mesh_phantom, complex_mesh_phantom)

                assert isinstance(result, MeshPhantom)
                assert result.parent is complex_mesh_phantom.parent
                assert result.children == complex_mesh_phantom.children
                assert len(result.tubes) == len(complex_mesh_phantom.tubes)
                assert mock_intersection.called


def test_meshes_tubes_clipping_handles_runtime_error(complex_mesh_phantom):
    clipping = MeshesTubesClipping()

    with patch(
        "trimesh.boolean.intersection",
        side_effect=RuntimeError("Boolean operation failed"),
    ):
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with pytest.raises(
                RuntimeError, match="Boolean operation failed for tubes clipping"
            ):
                clipping(complex_mesh_phantom, complex_mesh_phantom)


def test_meshes_tubes_clipping_validates_input_meshes(complex_mesh_phantom):
    clipping = MeshesTubesClipping()

    with patch(
        "magnet_pinn.generator.transforms._validate_input_meshes"
    ) as mock_validate:
        with patch(
            "trimesh.boolean.intersection", return_value=Mock(spec=trimesh.Trimesh)
        ):
            with patch("magnet_pinn.generator.transforms._validate_mesh"):
                mock_clipped = Mock(spec=trimesh.Trimesh)
                mock_clipped.vertices = np.array([[0, 0, 0]])
                mock_clipped.faces = np.array([[0, 1, 2]])
                mock_clipped.volume = 1.0
                mock_clipped.is_volume = True
                mock_clipped.remove_degenerate_faces = Mock()
                mock_clipped.remove_duplicate_faces = Mock()
                mock_clipped.remove_unreferenced_vertices = Mock()
                mock_clipped.fill_holes = Mock()

                with patch("trimesh.boolean.intersection", return_value=mock_clipped):
                    clipping(complex_mesh_phantom, complex_mesh_phantom)

                assert mock_validate.call_count == 2


def test_meshes_children_cutout_successful_operation(complex_mesh_phantom):
    cutout = MeshesChildrenCutout()

    mock_cutout_child = Mock(spec=trimesh.Trimesh)
    mock_cutout_child.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_cutout_child.faces = np.array([[0, 1, 2]])
    mock_cutout_child.volume = 1.0
    mock_cutout_child.is_volume = True
    mock_cutout_child.remove_degenerate_faces = Mock()
    mock_cutout_child.remove_duplicate_faces = Mock()
    mock_cutout_child.remove_unreferenced_vertices = Mock()
    mock_cutout_child.fill_holes = Mock()

    with patch(
        "trimesh.boolean.difference", return_value=mock_cutout_child
    ) as mock_difference:
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with patch("magnet_pinn.generator.transforms._validate_mesh"):
                result = cutout(complex_mesh_phantom, complex_mesh_phantom)

                assert isinstance(result, MeshPhantom)
                assert result.parent is complex_mesh_phantom.parent
                assert result.tubes == complex_mesh_phantom.tubes
                assert len(result.children) == len(complex_mesh_phantom.children)
                assert mock_difference.called


def test_meshes_children_cutout_handles_runtime_error(complex_mesh_phantom):
    cutout = MeshesChildrenCutout()

    with patch(
        "trimesh.boolean.difference",
        side_effect=RuntimeError("Boolean operation failed"),
    ):
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with pytest.raises(
                RuntimeError, match="Boolean operation failed for children cutout"
            ):
                cutout(complex_mesh_phantom, complex_mesh_phantom)


def test_meshes_parent_cutout_with_children_successful_operation(complex_mesh_phantom):
    cutout = MeshesParentCutoutWithChildren()

    mock_parent = Mock(spec=trimesh.Trimesh)
    mock_parent.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_parent.faces = np.array([[0, 1, 2]])
    mock_parent.volume = 1.0
    mock_parent.is_volume = True
    mock_parent.remove_degenerate_faces = Mock()
    mock_parent.remove_duplicate_faces = Mock()
    mock_parent.remove_unreferenced_vertices = Mock()
    mock_parent.fill_holes = Mock()

    with patch(
        "trimesh.boolean.difference", return_value=mock_parent
    ) as mock_difference:
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with patch("magnet_pinn.generator.transforms._validate_mesh"):
                result = cutout(complex_mesh_phantom, complex_mesh_phantom)

                assert isinstance(result, MeshPhantom)
                assert result.parent is mock_parent
                assert result.children == complex_mesh_phantom.children
                assert result.tubes == complex_mesh_phantom.tubes
                assert mock_difference.called


def test_meshes_parent_cutout_with_children_handles_runtime_error(complex_mesh_phantom):
    cutout = MeshesParentCutoutWithChildren()

    with patch(
        "trimesh.boolean.difference",
        side_effect=RuntimeError("Boolean operation failed"),
    ):
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with pytest.raises(
                RuntimeError,
                match="Boolean operation failed for parent cutout with children",
            ):
                cutout(complex_mesh_phantom, complex_mesh_phantom)


def test_meshes_parent_cutout_with_tubes_successful_operation(complex_mesh_phantom):
    cutout = MeshesParentCutoutWithTubes()

    mock_parent = Mock(spec=trimesh.Trimesh)
    mock_parent.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_parent.faces = np.array([[0, 1, 2]])
    mock_parent.volume = 1.0
    mock_parent.is_volume = True
    mock_parent.remove_degenerate_faces = Mock()
    mock_parent.remove_duplicate_faces = Mock()
    mock_parent.remove_unreferenced_vertices = Mock()
    mock_parent.fill_holes = Mock()

    with patch(
        "trimesh.boolean.difference", return_value=mock_parent
    ) as mock_difference:
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with patch("magnet_pinn.generator.transforms._validate_mesh"):
                result = cutout(complex_mesh_phantom, complex_mesh_phantom)

                assert isinstance(result, MeshPhantom)
                assert result.parent is mock_parent
                assert result.children == complex_mesh_phantom.children
                assert result.tubes == complex_mesh_phantom.tubes
                assert mock_difference.called


def test_meshes_parent_cutout_with_tubes_handles_runtime_error(complex_mesh_phantom):
    cutout = MeshesParentCutoutWithTubes()

    with patch(
        "trimesh.boolean.difference",
        side_effect=RuntimeError("Boolean operation failed"),
    ):
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with pytest.raises(
                RuntimeError,
                match="Boolean operation failed for parent cutout with tubes",
            ):
                cutout(complex_mesh_phantom, complex_mesh_phantom)


def test_meshes_children_clipping_successful_operation(complex_mesh_phantom):
    clipping = MeshesChildrenClipping()

    mock_clipped_child = Mock(spec=trimesh.Trimesh)
    mock_clipped_child.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_clipped_child.faces = np.array([[0, 1, 2]])
    mock_clipped_child.volume = 1.0
    mock_clipped_child.is_volume = True
    mock_clipped_child.remove_degenerate_faces = Mock()
    mock_clipped_child.remove_duplicate_faces = Mock()
    mock_clipped_child.remove_unreferenced_vertices = Mock()
    mock_clipped_child.fill_holes = Mock()

    with patch(
        "trimesh.boolean.intersection", return_value=mock_clipped_child
    ) as mock_intersection:
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with patch("magnet_pinn.generator.transforms._validate_mesh"):
                result = clipping(complex_mesh_phantom, complex_mesh_phantom)

                assert isinstance(result, MeshPhantom)
                assert result.parent is complex_mesh_phantom.parent
                assert result.tubes == complex_mesh_phantom.tubes
                assert len(result.children) == len(complex_mesh_phantom.children)
                assert mock_intersection.called


def test_meshes_children_clipping_handles_runtime_error(complex_mesh_phantom):
    clipping = MeshesChildrenClipping()

    with patch(
        "trimesh.boolean.intersection",
        side_effect=RuntimeError("Boolean operation failed"),
    ):
        with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
            with pytest.raises(
                RuntimeError, match="Boolean operation failed for children clipping"
            ):
                clipping(complex_mesh_phantom, complex_mesh_phantom)


def test_transform_pipeline_composition():
    """Test that transforms can be composed into pipelines."""
    transforms = [ToMesh(), MeshesCleaning(), MeshesRemesh(max_len=10.0)]

    # Test fixture: list contains Transform subtypes
    pipeline = Compose(transforms)  # type: ignore[arg-type]
    assert len(pipeline.transforms) == 3
    assert isinstance(pipeline.transforms[0], ToMesh)
    assert isinstance(pipeline.transforms[1], MeshesCleaning)
    assert isinstance(pipeline.transforms[2], MeshesRemesh)


def test_boolean_operations_error_handling():
    """Test that all boolean operations handle errors consistently."""
    parent_mock = Mock(spec=trimesh.Trimesh)
    parent_mock.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    parent_mock.faces = np.array([[0, 1, 2]])
    parent_mock.volume = 1.0

    child_mock = Mock(spec=trimesh.Trimesh)
    child_mock.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    child_mock.faces = np.array([[0, 1, 2]])
    child_mock.volume = 0.5

    tube_mock = Mock(spec=trimesh.Trimesh)
    tube_mock.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    tube_mock.faces = np.array([[0, 1, 2]])
    tube_mock.volume = 0.3

    phantom = MeshPhantom(parent=parent_mock, children=[child_mock], tubes=[tube_mock])

    transforms = [
        MeshesTubesClipping(),
        MeshesChildrenCutout(),
        MeshesParentCutoutWithChildren(),
        MeshesParentCutoutWithTubes(),
        MeshesChildrenClipping(),
    ]

    for transform in transforms:
        with patch(
            "trimesh.boolean.intersection", side_effect=RuntimeError("Test error")
        ):
            with patch(
                "trimesh.boolean.difference", side_effect=RuntimeError("Test error")
            ):
                with patch("magnet_pinn.generator.transforms._validate_input_meshes"):
                    with pytest.raises(RuntimeError, match="Boolean operation failed"):
                        transform(phantom, phantom)


def test_mesh_cleaning_operations_applied():
    cleaning = MeshesCleaning()
    mesh = Mock(spec=trimesh.Trimesh)
    mesh.copy.return_value = mesh
    mesh.nondegenerate_faces.return_value = np.array([0, 1, 2])
    mesh.unique_faces.return_value = np.array([0, 1, 2])

    with patch("trimesh.repair.fill_holes"):
        with patch("trimesh.repair.fix_normals"):
            cleaning._clean_mesh(mesh)

            mesh.update_faces.assert_called()
            mesh.merge_vertices.assert_called()
            mesh.fix_normals.assert_called()
            mesh.remove_unreferenced_vertices.assert_called()


def test_remesh_subdivide_parameters():
    """Test that remesh passes correct parameters to subdivide function."""
    remesh = MeshesRemesh(max_len=3.5)
    mesh = Mock(spec=trimesh.Trimesh)
    mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mesh.faces = np.array([[0, 1, 2]])

    with patch("trimesh.remesh.subdivide_to_size") as mock_subdivide:
        mock_subdivide.return_value = (mesh.vertices, mesh.faces)

        remesh._remesh(mesh)

        mock_subdivide.assert_called_once_with(mesh.vertices, mesh.faces, max_edge=3.5)


def test_meshes_tubes_clipping_returns_early_when_no_tubes():
    clipping = MeshesTubesClipping()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    phantom = MeshPhantom(parent=parent_mesh, children=[], tubes=[])

    result = clipping(phantom, phantom)

    assert result is phantom
    assert len(result.tubes) == 0


def test_meshes_children_cutout_returns_early_when_no_tubes():
    cutout = MeshesChildrenCutout()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    child_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    phantom = MeshPhantom(parent=parent_mesh, children=[child_mesh], tubes=[])

    result = cutout(phantom, phantom)

    assert result is phantom
    assert len(result.tubes) == 0
    assert len(result.children) == 1


def test_meshes_children_cutout_returns_early_when_no_children():
    cutout = MeshesChildrenCutout()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    tube_mesh = trimesh.creation.cylinder(radius=0.3, height=1.0)
    phantom = MeshPhantom(parent=parent_mesh, children=[], tubes=[tube_mesh])

    result = cutout(phantom, phantom)

    assert result is phantom
    assert len(result.tubes) == 1
    assert len(result.children) == 0


def test_meshes_parent_cutout_with_children_returns_early_when_no_children():
    cutout = MeshesParentCutoutWithChildren()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    phantom = MeshPhantom(parent=parent_mesh, children=[], tubes=[])

    result = cutout(phantom, phantom)

    assert result is phantom
    assert len(result.children) == 0


def test_meshes_parent_cutout_with_tubes_returns_early_when_no_tubes():
    cutout = MeshesParentCutoutWithTubes()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    phantom = MeshPhantom(parent=parent_mesh, children=[], tubes=[])

    result = cutout(phantom, phantom)

    assert result is phantom
    assert len(result.tubes) == 0


def test_meshes_children_clipping_returns_early_when_no_children():
    clipping = MeshesChildrenClipping()
    parent_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    phantom = MeshPhantom(parent=parent_mesh, children=[], tubes=[])

    result = clipping(phantom, phantom)

    assert result is phantom
    assert len(result.children) == 0
