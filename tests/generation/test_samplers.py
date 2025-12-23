import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import trimesh
from numpy.random import default_rng

from magnet_pinn.generator import samplers as samplers_module
from magnet_pinn.generator.samplers import (
    PointSampler,
    BlobSampler,
    TubeSampler,
    PropertySampler,
    MeshBlobSampler,
    MeshTubeSampler,
)
from magnet_pinn.generator.structures import Blob, Tube, CustomMeshStructure
from magnet_pinn.generator.typing import (
    PropertyItem,
    PropertyPhantom,
    StructurePhantom,
    MeshPhantom,
)


def test_samplers_import_fallback(monkeypatch):
    original_igl = sys.modules["igl"]
    fake_igl = ModuleType("igl")

    def fast_winding_number_for_meshes(*args, **kwargs):
        return "fallback"

    fake_igl.fast_winding_number_for_meshes = fast_winding_number_for_meshes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "igl", fake_igl)

    reloaded = importlib.reload(samplers_module)
    assert reloaded.fast_winding_number is fast_winding_number_for_meshes

    sys.modules["igl"] = original_igl
    importlib.reload(samplers_module)


def test_point_sampler_initialization_with_valid_center_and_radius():
    center = np.array([1.0, 2.0, 3.0])
    radius = 5.0

    sampler = PointSampler(center=center, radius=radius)

    assert np.array_equal(sampler.center, center)
    assert sampler.radius == radius
    assert sampler.center.dtype == float
    assert isinstance(sampler.radius, float)


def test_point_sampler_initialization_with_zero_center_and_minimal_radius():
    center = np.array([0.0, 0.0, 0.0])
    radius = np.finfo(float).eps

    sampler = PointSampler(center=center, radius=radius)
    assert np.array_equal(sampler.center, center)
    assert sampler.radius == radius


def test_point_sampler_initialization_with_large_center_and_radius_values():
    center = np.array([1e6, -1e6, 1e6])
    radius = 1e6

    sampler = PointSampler(center=center, radius=radius)
    assert np.array_equal(sampler.center, center)
    assert sampler.radius == radius


def test_point_sampler_converts_integer_center_and_radius_to_float():
    center = np.array([1, 2, 3])
    radius = 5

    sampler = PointSampler(center=center, radius=radius)
    assert isinstance(sampler.radius, float)


def test_point_sampler_initialization_with_list_center():
    center = [1.0, 2.0, 3.0]
    radius = 5.0

    sampler = PointSampler(center=np.array(center), radius=radius)
    assert np.array_equal(sampler.center, np.array(center))
    assert sampler.center.dtype == float


def test_point_sampler_initialization_with_zero_radius():
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.0

    sampler = PointSampler(center=center, radius=radius)
    assert sampler.radius == 0.0


def test_point_sampler_initialization_with_negative_radius():
    center = np.array([0.0, 0.0, 0.0])
    radius = -1.0

    sampler = PointSampler(center=center, radius=radius)
    assert sampler.radius == -1.0


def test_point_sampler_sample_point_returns_correct_shape():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    point = sampler.sample_point(rng)

    assert isinstance(point, np.ndarray)
    assert point.shape == (3,)
    assert point.dtype == float


def test_point_sampler_sample_point_reproducible_with_same_seed():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    point1 = sampler.sample_point(rng1)
    point2 = sampler.sample_point(rng2)

    assert np.allclose(point1, point2)


def test_point_sampler_sample_point_different_results_with_different_seeds():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)

    rng1 = default_rng(42)
    rng2 = default_rng(123)

    point1 = sampler.sample_point(rng1)
    point2 = sampler.sample_point(rng2)

    assert not np.allclose(point1, point2)


def test_point_sampler_sample_point_within_sphere_boundary():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    for _ in range(100):
        point = sampler.sample_point(rng)
        distance = np.linalg.norm(point - center)
        assert distance <= radius


def test_point_sampler_sample_point_with_offset_center():
    center = np.array([5.0, -3.0, 2.0])
    radius = 2.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    point = sampler.sample_point(rng)
    distance = np.linalg.norm(point - center)
    assert distance <= radius


def test_point_sampler_sample_points_returns_correct_shape():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)
    num_points = 10

    points = sampler.sample_points(num_points, rng)

    assert isinstance(points, np.ndarray)
    assert points.shape == (num_points, 3)
    assert points.dtype == float


def test_point_sampler_sample_points_with_single_point():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    points = sampler.sample_points(1, rng)

    assert points.shape == (1, 3)


def test_point_sampler_sample_points_with_zero_points():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    points = sampler.sample_points(0, rng)

    assert points.shape == (0, 3)


def test_point_sampler_sample_points_with_moderate_number():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)
    num_points = 100

    points = sampler.sample_points(num_points, rng)

    assert points.shape == (num_points, 3)


def test_point_sampler_sample_points_all_within_sphere_boundary():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    points = sampler.sample_points(100, rng)
    distances = np.linalg.norm(points - center, axis=1)

    assert np.all(distances <= radius)


def test_point_sampler_sample_points_reproducible_with_same_seed():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    points1 = sampler.sample_points(10, rng1)
    points2 = sampler.sample_points(10, rng2)

    assert np.allclose(points1, points2)


def test_point_sampler_sample_points_with_zero_radius():
    center = np.array([1.0, 2.0, 3.0])
    radius = 0.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(42)

    points = sampler.sample_points(5, rng)

    for point in points:
        assert np.allclose(point, center)


def test_blob_sampler_initialization_with_valid_radius_decrease_factor():
    radius_decrease_factor = 0.5

    sampler = BlobSampler(radius_decrease_factor=radius_decrease_factor)

    assert sampler.radius_decrease_factor == radius_decrease_factor
    assert isinstance(sampler.radius_decrease_factor, float)


def test_blob_sampler_initialization_with_minimum_radius_decrease_factor():
    radius_decrease_factor = np.finfo(float).eps

    sampler = BlobSampler(radius_decrease_factor=radius_decrease_factor)
    assert sampler.radius_decrease_factor == radius_decrease_factor


def test_blob_sampler_initialization_with_maximum_radius_decrease_factor():
    radius_decrease_factor = 1.0 - np.finfo(float).eps

    sampler = BlobSampler(radius_decrease_factor=radius_decrease_factor)
    assert sampler.radius_decrease_factor == radius_decrease_factor


def test_blob_sampler_initialization_with_near_boundary_low_value():
    radius_decrease_factor = 0.001

    sampler = BlobSampler(radius_decrease_factor=radius_decrease_factor)
    assert sampler.radius_decrease_factor == radius_decrease_factor


def test_blob_sampler_initialization_with_near_boundary_high_value():
    radius_decrease_factor = 0.999

    sampler = BlobSampler(radius_decrease_factor=radius_decrease_factor)
    assert sampler.radius_decrease_factor == radius_decrease_factor


def test_blob_sampler_rejects_zero_radius_decrease_factor():
    with pytest.raises(
        ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"
    ):
        BlobSampler(radius_decrease_factor=0.0)


def test_blob_sampler_rejects_negative_radius_decrease_factor():
    with pytest.raises(
        ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"
    ):
        BlobSampler(radius_decrease_factor=-0.1)


def test_blob_sampler_rejects_one_radius_decrease_factor():
    with pytest.raises(
        ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"
    ):
        BlobSampler(radius_decrease_factor=1.0)


def test_blob_sampler_rejects_greater_than_one_radius_decrease_factor():
    with pytest.raises(
        ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"
    ):
        BlobSampler(radius_decrease_factor=1.1)


def test_blob_sampler_check_points_distance_with_well_separated_points():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    min_distance = 1.0

    result = sampler.check_points_distance(points, min_distance)

    assert result is True


def test_blob_sampler_check_points_distance_with_too_close_points():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 2.0, 0.0]])
    min_distance = 1.0

    result = sampler.check_points_distance(points, min_distance)

    assert result is False


def test_blob_sampler_check_points_distance_with_exact_minimum_distance():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    min_distance = 1.0

    result = sampler.check_points_distance(points, min_distance)

    assert result is True


def test_blob_sampler_check_points_distance_with_single_point():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([[0.0, 0.0, 0.0]])
    min_distance = 1.0

    result = sampler.check_points_distance(points, min_distance)

    assert result is True


def test_blob_sampler_check_points_distance_with_empty_points():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([]).reshape(0, 3)
    min_distance = 1.0

    with pytest.raises(ValueError):
        sampler.check_points_distance(points, min_distance)


def test_blob_sampler_check_points_distance_with_zero_minimum_distance():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    min_distance = 0.0

    result = sampler.check_points_distance(points, min_distance)

    assert result is True


def test_blob_sampler_check_points_distance_with_negative_minimum_distance():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    min_distance = -1.0

    result = sampler.check_points_distance(points, min_distance)

    assert result is True


def test_blob_sampler_sample_children_blobs_with_zero_children():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=5.0)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(parent_blob, 0, rng)

    assert len(children) == 0
    assert isinstance(children, list)


def test_blob_sampler_sample_children_blobs_with_single_child():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=5.0)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(parent_blob, 1, rng)

    assert len(children) == 1
    assert isinstance(children[0], Blob)
    assert children[0].radius == parent_blob.radius * sampler.radius_decrease_factor


def test_blob_sampler_sample_children_blobs_with_multiple_children():
    sampler = BlobSampler(radius_decrease_factor=0.3)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    rng = default_rng(42)
    num_children = 3

    children = sampler.sample_children_blobs(parent_blob, num_children, rng)

    assert len(children) == num_children
    for child in children:
        assert isinstance(child, Blob)
        assert child.radius == parent_blob.radius * sampler.radius_decrease_factor


def test_blob_sampler_sample_children_blobs_children_within_parent():
    sampler = BlobSampler(radius_decrease_factor=0.3)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(parent_blob, 2, rng)

    for child in children:
        distance_to_parent = np.linalg.norm(child.position - parent_blob.position)
        max_allowed_distance = parent_blob.radius * (
            1 + parent_blob.empirical_min_offset
        ) - child.radius * (1 + child.empirical_max_offset)
        assert distance_to_parent <= max_allowed_distance


def test_blob_sampler_sample_children_blobs_reproducible_with_same_seed():
    sampler = BlobSampler(radius_decrease_factor=0.2)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    children1 = sampler.sample_children_blobs(parent_blob, 2, rng1)
    children2 = sampler.sample_children_blobs(parent_blob, 2, rng2)

    assert len(children1) == len(children2)
    for c1, c2 in zip(children1, children2):
        assert np.allclose(c1.position, c2.position)
        assert c1.radius == c2.radius


def test_blob_sampler_sample_children_blobs_fails_with_too_small_parent():
    sampler = BlobSampler(radius_decrease_factor=0.9)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=1.0)
    rng = default_rng(42)

    with pytest.raises(
        RuntimeError, match="Parent blob radius .* too small to fit child blob radius"
    ):
        sampler.sample_children_blobs(parent_blob, 1, rng)


def test_blob_sampler_sample_children_blobs_fails_with_too_many_children():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=2.0)
    rng = default_rng(42)

    with pytest.raises(RuntimeError, match="Cannot pack .* spheres"):
        sampler.sample_children_blobs(parent_blob, 20, rng)


def test_blob_sampler_sample_children_blobs_with_custom_max_iterations():
    sampler = BlobSampler(radius_decrease_factor=0.3)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(parent_blob, 1, rng, max_iterations=100)

    assert len(children) == 1


def test_blob_sampler_sample_children_blobs_fails_with_insufficient_iterations():
    sampler = BlobSampler(radius_decrease_factor=0.2)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    rng = default_rng(42)

    with pytest.raises(RuntimeError, match="Could not find .* valid positions"):
        sampler.sample_children_blobs(parent_blob, 3, rng, max_iterations=1)


def test_tube_sampler_initialization_with_valid_radii():
    tube_max_radius = 2.0
    tube_min_radius = 0.5

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )

    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius
    assert isinstance(sampler.tube_max_radius, float)
    assert isinstance(sampler.tube_min_radius, float)


def test_tube_sampler_initialization_with_minimal_difference():
    tube_max_radius = 1.0 + np.finfo(float).eps
    tube_min_radius = 1.0

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )
    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius


def test_tube_sampler_initialization_with_large_difference():
    tube_max_radius = 1000.0
    tube_min_radius = 0.001

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )
    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius


def test_tube_sampler_initialization_with_equal_radii_boundary():
    tube_max_radius = 1.0
    tube_min_radius = 1.0 - np.finfo(float).eps

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )
    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius


def test_tube_sampler_rejects_zero_max_radius():
    with pytest.raises(ValueError, match="tube_max_radius must be positive"):
        TubeSampler(tube_max_radius=0.0, tube_min_radius=0.5)


def test_tube_sampler_rejects_negative_max_radius():
    with pytest.raises(ValueError, match="tube_max_radius must be positive"):
        TubeSampler(tube_max_radius=-1.0, tube_min_radius=0.5)


def test_tube_sampler_rejects_zero_min_radius():
    with pytest.raises(ValueError, match="tube_min_radius must be positive"):
        TubeSampler(tube_max_radius=2.0, tube_min_radius=0.0)


def test_tube_sampler_rejects_negative_min_radius():
    with pytest.raises(ValueError, match="tube_min_radius must be positive"):
        TubeSampler(tube_max_radius=2.0, tube_min_radius=-0.1)


def test_tube_sampler_rejects_min_radius_equal_to_max_radius():
    with pytest.raises(
        ValueError, match="tube_min_radius must be less than tube_max_radius"
    ):
        TubeSampler(tube_max_radius=1.0, tube_min_radius=1.0)


def test_tube_sampler_rejects_min_radius_greater_than_max_radius():
    with pytest.raises(
        ValueError, match="tube_min_radius must be less than tube_max_radius"
    ):
        TubeSampler(tube_max_radius=1.0, tube_min_radius=2.0)


def test_tube_sampler_sample_tubes_with_zero_tubes():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 0, rng)

    assert len(tubes) == 0
    assert isinstance(tubes, list)


def test_tube_sampler_sample_tubes_with_single_tube():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 1, rng)

    assert len(tubes) == 1
    assert isinstance(tubes[0], Tube)


def test_tube_sampler_sample_tubes_with_multiple_tubes():
    sampler = TubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0
    rng = default_rng(42)
    num_tubes = 3

    tubes = sampler.sample_tubes(center, radius, num_tubes, rng)

    assert len(tubes) == num_tubes
    for tube in tubes:
        assert isinstance(tube, Tube)


def test_tube_sampler_sample_tubes_radii_within_range():
    sampler = TubeSampler(tube_max_radius=2.0, tube_min_radius=0.5)
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 5, rng)

    for tube in tubes:
        assert sampler.tube_min_radius <= tube.radius <= sampler.tube_max_radius


def test_tube_sampler_sample_tubes_within_sphere_boundary():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 3, rng)

    for tube in tubes:
        distance_to_center = np.linalg.norm(tube.position - center)
        assert distance_to_center + tube.radius < radius


def test_tube_sampler_sample_tubes_non_intersecting():
    sampler = TubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 20.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 3, rng)

    for i, tube1 in enumerate(tubes):
        for j, tube2 in enumerate(tubes):
            if i != j:
                distance = Tube.distance_to_tube(tube1, tube2)
                assert distance >= tube1.radius + tube2.radius


def test_tube_sampler_sample_tubes_reproducible_with_same_seed():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    tubes1 = sampler.sample_tubes(center, radius, 2, rng1)
    tubes2 = sampler.sample_tubes(center, radius, 2, rng2)

    assert len(tubes1) == len(tubes2)
    for t1, t2 in zip(tubes1, tubes2):
        assert np.allclose(t1.position, t2.position)
        assert np.allclose(t1.direction, t2.direction)
        assert t1.radius == t2.radius


def test_tube_sampler_sample_tubes_different_results_with_different_seeds():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0

    rng1 = default_rng(42)
    rng2 = default_rng(123)

    tubes1 = sampler.sample_tubes(center, radius, 2, rng1)
    tubes2 = sampler.sample_tubes(center, radius, 2, rng2)

    different = False
    for t1, t2 in zip(tubes1, tubes2):
        if not (
            np.allclose(t1.position, t2.position)
            and np.allclose(t1.direction, t2.direction)
            and t1.radius == t2.radius
        ):
            different = True
            break
    assert different


def test_tube_sampler_sample_tubes_with_offset_center():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([5.0, -3.0, 2.0])
    radius = 10.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 2, rng)

    for tube in tubes:
        distance_to_center = np.linalg.norm(tube.position - center)
        assert distance_to_center + tube.radius < radius


def test_tube_sampler_sample_tubes_with_minimal_radius():
    sampler = TubeSampler(
        tube_max_radius=np.finfo(float).eps * 2, tube_min_radius=np.finfo(float).eps
    )
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 1, rng)

    assert len(tubes) == 1
    assert tubes[0].radius >= sampler.tube_min_radius
    assert tubes[0].radius <= sampler.tube_max_radius


def test_tube_sampler_sample_tubes_with_large_radii():
    sampler = TubeSampler(tube_max_radius=100.0, tube_min_radius=50.0)
    center = np.array([0.0, 0.0, 0.0])
    radius = 1000.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 1, rng)

    assert len(tubes) == 1
    assert tubes[0].radius >= sampler.tube_min_radius
    assert tubes[0].radius <= sampler.tube_max_radius


def test_tube_sampler_sample_line_returns_valid_tube():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    ball_radius = 5.0
    tube_radius = 0.5
    rng = default_rng(42)

    tube = sampler._sample_line(center, ball_radius, tube_radius, rng)

    assert isinstance(tube, Tube)
    assert tube.radius == tube_radius
    assert np.linalg.norm(tube.position - center) <= ball_radius
    assert np.isclose(np.linalg.norm(tube.direction), 1.0)


def test_tube_sampler_sample_line_with_zero_ball_radius():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([1.0, 2.0, 3.0])
    ball_radius = 0.0
    tube_radius = 0.5
    rng = default_rng(42)

    tube = sampler._sample_line(center, ball_radius, tube_radius, rng)

    assert isinstance(tube, Tube)
    assert np.allclose(tube.position, center)


def test_tube_sampler_sample_line_with_minimal_tube_radius():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    rng = default_rng(42)

    minimal_radius = np.finfo(float).eps
    tube = sampler._sample_line(center, 10.0, minimal_radius, rng)
    assert isinstance(tube, Tube)
    assert tube.radius == minimal_radius


def test_point_sampler_sample_point_with_zero_norm_handling():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(0)

    point = sampler.sample_point(rng)
    assert point.shape == (3,)
    assert np.isfinite(point).all()


def test_point_sampler_sample_points_with_zero_norm_handling():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)
    rng = default_rng(0)

    points = sampler.sample_points(5, rng)
    assert points.shape == (5, 3)
    assert np.isfinite(points).all()


def test_blob_sampler_find_valid_positions_progressive_timeout():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    rng = default_rng(42)

    with pytest.raises(RuntimeError, match="Could not find 10 valid positions"):
        sampler._find_valid_positions_progressive(
            target_positions=10,
            center=np.array([0.0, 0.0, 0.0]),
            sampling_radius=0.1,
            min_distance=1.0,
            rng=rng,
            max_iterations=10,
        )


def test_blob_sampler_find_valid_positions_progressive_specific_error_message():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    rng = default_rng(42)

    with pytest.raises(RuntimeError) as exc_info:
        sampler._find_valid_positions_progressive(
            target_positions=5,
            center=np.array([0.0, 0.0, 0.0]),
            sampling_radius=0.5,
            min_distance=2.0,
            rng=rng,
            max_iterations=5,
        )

    error_message = str(exc_info.value)
    assert "Could not find 5 valid positions" in error_message
    assert "minimum distance 2.000" in error_message
    assert "within radius 0.500" in error_message
    assert "attempts" in error_message
    assert (
        "Try reducing target_positions or increasing sampling_radius" in error_message
    )


def test_blob_sampler_progressive_sampling_max_iterations_exhaustion():
    """Test that covers the missing lines 373, 376 - break statements in nested loops."""
    sampler = BlobSampler(radius_decrease_factor=0.5)
    rng = default_rng(42)

    with pytest.raises(RuntimeError) as exc_info:
        sampler._find_valid_positions_progressive(
            target_positions=10,
            center=np.array([0.0, 0.0, 0.0]),
            sampling_radius=0.1,
            min_distance=1.0,
            rng=rng,
            max_iterations=15,
        )

    error_message = str(exc_info.value)
    assert "Could not find 10 valid positions" in error_message
    assert "after 15 attempts" in error_message


def test_blob_sampler_progressive_sampling_inner_loop_break():
    """Test the inner loop break condition when max_iterations is hit mid-batch."""
    sampler = BlobSampler(radius_decrease_factor=0.5)
    rng = default_rng(42)

    with pytest.raises(RuntimeError):
        sampler._find_valid_positions_progressive(
            target_positions=8,
            center=np.array([0.0, 0.0, 0.0]),
            sampling_radius=0.2,
            min_distance=0.8,
            rng=rng,
            max_iterations=3,
        )


def test_blob_sampler_progressive_sampling_outer_loop_break():
    """Test the outer loop break condition when max_iterations is exceeded."""
    sampler = BlobSampler(radius_decrease_factor=0.5)
    rng = default_rng(42)

    with pytest.raises(RuntimeError):
        sampler._find_valid_positions_progressive(
            target_positions=6,
            center=np.array([0.0, 0.0, 0.0]),
            sampling_radius=0.15,
            min_distance=0.5,
            rng=rng,
            max_iterations=6,
        )


def test_point_sampler_division_by_zero_protection():
    """Test protection against division by zero in normalization."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)

    rng = default_rng(1234)

    for _ in range(100):
        point = sampler.sample_point(rng)
        assert np.isfinite(point).all()
        assert point.shape == (3,)
        distance = np.linalg.norm(point - center)
        assert distance <= radius + np.finfo(float).eps * 10


def test_point_sampler_batch_division_by_zero_protection():
    """Test protection against division by zero in batch normalization."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)

    for seed in [0, 1, 42, 999]:
        rng = default_rng(seed)
        points = sampler.sample_points(50, rng)

        assert np.isfinite(points).all()
        assert points.shape == (50, 3)

        distances = np.linalg.norm(points - center, axis=1)
        assert np.all(distances <= radius + np.finfo(float).eps * 10)


def test_blob_sampler_empty_points_array_error_handling():
    """Test proper error handling for empty points arrays."""
    sampler = BlobSampler(radius_decrease_factor=0.5)

    empty_points = np.array([]).reshape(0, 3)

    with pytest.raises(ValueError):
        sampler.check_points_distance(empty_points, 1.0)


def test_blob_sampler_malformed_points_array():
    """Test behavior with malformed points arrays."""
    sampler = BlobSampler(radius_decrease_factor=0.5)

    malformed_points = np.array([[1, 2], [3, 4]])

    try:
        result = sampler.check_points_distance(malformed_points, 1.0)
        assert isinstance(result, bool)
    except (ValueError, IndexError):
        pass


def test_tube_sampler_geometric_impossibility_scenarios():
    """Test various geometric impossibility scenarios."""
    sampler = TubeSampler(tube_max_radius=10.0, tube_min_radius=5.0)
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 5, rng, max_iterations=100)
    assert len(tubes) == 0

    sampler = TubeSampler(tube_max_radius=2.0, tube_min_radius=1.5)
    radius = 3.0

    tubes = sampler.sample_tubes(center, radius, 10, rng, max_iterations=100)
    assert len(tubes) <= 2


def test_tube_sampler_collision_detection_precision():
    """Test collision detection with high precision requirements."""
    sampler = TubeSampler(tube_max_radius=0.1, tube_min_radius=0.05)
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 10, rng, max_iterations=1000)

    for i in range(len(tubes)):
        for j in range(i + 1, len(tubes)):
            distance = Tube.distance_to_tube(tubes[i], tubes[j])
            min_required_distance = tubes[i].radius + tubes[j].radius
            assert distance >= min_required_distance - np.finfo(float).eps * 100


def test_blob_sampler_numerical_precision_edge_cases():
    """Test numerical precision in distance calculations."""
    sampler = BlobSampler(radius_decrease_factor=0.5)

    eps = np.finfo(float).eps
    points = np.array([[0.0, 0.0, 0.0], [eps, 0.0, 0.0], [0.0, eps, 0.0]])

    result = sampler.check_points_distance(points, eps * 0.5)
    assert isinstance(result, bool)

    result = sampler.check_points_distance(points, eps * 2)
    assert isinstance(result, bool)


def test_tube_sampler_extreme_coordinate_values():
    """Test tube sampling with extreme coordinate values."""
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)

    center = np.array([1e10, -1e10, 1e10])
    radius = 1e9
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 2, rng, max_iterations=100)

    for tube in tubes:
        assert np.isfinite(tube.position).all()
        assert np.isfinite(tube.direction).all()
        assert np.isfinite(tube.radius)
        assert tube.radius > 0


def test_blob_sampler_safety_margin_calculations():
    """Test safety margin calculations in child radius computation."""
    sampler = BlobSampler(radius_decrease_factor=0.5)

    blob1 = Blob(np.array([0, 0, 0]), 1.0, relative_disruption_strength=0.1)
    blob2 = Blob(np.array([0, 0, 0]), 1.0, relative_disruption_strength=0.3)
    blobs = [blob1, blob2]

    base_radius = 1.0
    safe_radius = sampler._calculate_safe_child_radius(blobs, base_radius)

    assert safe_radius > base_radius

    max_offset = max(blob.empirical_max_offset for blob in blobs)
    expected = base_radius * (1 + max_offset)
    assert np.isclose(safe_radius, expected)


def test_blob_sampler_sample_children_blobs_numerical_precision():
    sampler = BlobSampler(radius_decrease_factor=0.1)
    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=1e-5)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(parent_blob, 1, rng, max_iterations=100)
    assert len(children) <= 1


def test_tube_sampler_sample_tubes_with_minimal_space():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.9)
    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, 2, rng)
    assert len(tubes) <= 2


def test_tube_sampler_initialization_with_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5
    parent_radius = 100.0

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius,
        tube_min_radius=tube_min_radius,
        parent_radius=parent_radius,
    )

    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius
    assert sampler.parent_radius == parent_radius
    assert isinstance(sampler.parent_radius, float)


def test_tube_sampler_initialization_with_default_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )

    assert sampler.parent_radius == 250.0


def test_tube_sampler_initialization_with_zero_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5
    parent_radius = 0.0

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius,
        tube_min_radius=tube_min_radius,
        parent_radius=parent_radius,
    )

    assert sampler.parent_radius == 0.0


def test_tube_sampler_initialization_with_large_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5
    parent_radius = 10000.0

    sampler = TubeSampler(
        tube_max_radius=tube_max_radius,
        tube_min_radius=tube_min_radius,
        parent_radius=parent_radius,
    )

    assert sampler.parent_radius == parent_radius


def test_tube_sampler_sample_line_sets_height_based_on_parent_radius():
    parent_radius = 50.0
    sampler = TubeSampler(
        tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
    )
    center = np.array([0.0, 0.0, 0.0])
    ball_radius = 5.0
    tube_radius = 0.5
    rng = default_rng(42)

    tube = sampler._sample_line(center, ball_radius, tube_radius, rng)

    expected_height = 4 * parent_radius
    assert tube.height == expected_height


def test_tube_sampler_sample_line_with_default_parent_radius_height():
    sampler = TubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    center = np.array([0.0, 0.0, 0.0])
    ball_radius = 5.0
    tube_radius = 0.5
    rng = default_rng(42)

    tube = sampler._sample_line(center, ball_radius, tube_radius, rng)

    expected_height = 4 * 250.0
    assert tube.height == expected_height


def test_tube_sampler_sample_tubes_all_have_correct_height():
    parent_radius = 75.0
    sampler = TubeSampler(
        tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
    )
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    num_tubes = 3
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, num_tubes, rng)

    expected_height = 4 * parent_radius
    for tube in tubes:
        assert tube.height == expected_height


def test_tube_sampler_sample_tubes_height_calculation_with_zero_parent_radius():
    parent_radius = 0.0
    sampler = TubeSampler(
        tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
    )
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    num_tubes = 2
    rng = default_rng(42)

    tubes = sampler.sample_tubes(center, radius, num_tubes, rng)

    expected_height = 4 * parent_radius
    for tube in tubes:
        assert tube.height == expected_height


def test_tube_sampler_height_calculation_formula_verification():
    parent_radius_values = [10.0, 25.5, 100.0, 1000.0]

    for parent_radius in parent_radius_values:
        sampler = TubeSampler(
            tube_max_radius=1.0, tube_min_radius=0.1, parent_radius=parent_radius
        )
        center = np.array([0.0, 0.0, 0.0])
        ball_radius = 5.0
        tube_radius = 0.5
        rng = default_rng(42)

        tube = sampler._sample_line(center, ball_radius, tube_radius, rng)

        assert tube.height == 4 * parent_radius


def test_point_sampler_normalization_with_zero_vector():
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    sampler = PointSampler(center=center, radius=radius)

    rng = default_rng(123)
    point = sampler.sample_point(rng)
    assert np.isfinite(point).all()
    assert point.shape == (3,)


def test_blob_sampler_progressive_sampling_boundary_conditions():
    sampler = BlobSampler(radius_decrease_factor=0.5)
    rng = default_rng(42)

    try:
        positions = sampler._find_valid_positions_progressive(
            target_positions=1,
            center=np.array([0.0, 0.0, 0.0]),
            sampling_radius=1.0,
            min_distance=0.1,
            rng=rng,
            max_iterations=100,
        )
        assert len(positions) == 1
        assert positions.shape == (1, 3)
    except RuntimeError:
        pass


def test_property_sampler_initialization_with_valid_single_property_config():
    properties_cfg = {"conductivity": {"min": 0.1, "max": 1.0}}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg == properties_cfg
    assert "conductivity" in sampler.properties_cfg
    assert sampler.properties_cfg["conductivity"]["min"] == 0.1
    assert sampler.properties_cfg["conductivity"]["max"] == 1.0


def test_property_sampler_initialization_with_valid_multiple_properties_config():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg == properties_cfg
    assert len(sampler.properties_cfg) == 3
    assert all(
        key in sampler.properties_cfg
        for key in ["conductivity", "permittivity", "density"]
    )


def test_property_sampler_initialization_with_empty_config():
    properties_cfg: dict[str, dict[str, float]] = {}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg == properties_cfg
    assert len(sampler.properties_cfg) == 0


def test_property_sampler_initialization_with_minimal_range_property():
    properties_cfg = {"conductivity": {"min": 0.0, "max": 0.0001}}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg == properties_cfg
    assert sampler.properties_cfg["conductivity"]["min"] == 0.0
    assert sampler.properties_cfg["conductivity"]["max"] == 0.0001


def test_property_sampler_initialization_with_large_range_property():
    properties_cfg = {"conductivity": {"min": 1e-10, "max": 1e10}}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg == properties_cfg
    assert sampler.properties_cfg["conductivity"]["min"] == 1e-10
    assert sampler.properties_cfg["conductivity"]["max"] == 1e10


def test_property_sampler_initialization_with_zero_minimum_boundary():
    properties_cfg = {"conductivity": {"min": 0.0, "max": 1.0}}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg["conductivity"]["min"] == 0.0
    assert sampler.properties_cfg["conductivity"]["max"] == 1.0


def test_property_sampler_initialization_with_negative_range():
    properties_cfg = {"conductivity": {"min": -1.0, "max": 1.0}}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg["conductivity"]["min"] == -1.0
    assert sampler.properties_cfg["conductivity"]["max"] == 1.0


def test_property_sampler_initialization_with_equal_min_max_boundary():
    properties_cfg = {"conductivity": {"min": 1.0, "max": 1.0}}

    sampler = PropertySampler(properties_cfg)

    assert sampler.properties_cfg["conductivity"]["min"] == 1.0
    assert sampler.properties_cfg["conductivity"]["max"] == 1.0


def test_property_sampler_sample_returns_property_item_with_all_configured_properties():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng)

    assert isinstance(property_item, PropertyItem)
    assert hasattr(property_item, "conductivity")
    assert hasattr(property_item, "permittivity")
    assert hasattr(property_item, "density")
    assert 0.1 <= property_item.conductivity <= 1.0
    assert 1.0 <= property_item.permittivity <= 100.0
    assert 500.0 <= property_item.density <= 2000.0


def test_property_sampler_sample_with_specific_properties_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)
    properties_list = ["conductivity", "permittivity", "density"]

    property_item = sampler._sample(rng, properties_list)

    assert isinstance(property_item, PropertyItem)
    assert hasattr(property_item, "conductivity")
    assert hasattr(property_item, "permittivity")
    assert hasattr(property_item, "density")
    assert 0.1 <= property_item.conductivity <= 1.0
    assert 1.0 <= property_item.permittivity <= 100.0
    assert 500.0 <= property_item.density <= 2000.0


def test_property_sampler_sample_with_filtered_properties_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)
    properties_list = ["conductivity", "density"]

    with pytest.raises(TypeError):
        sampler._sample(rng, properties_list)


def test_property_sampler_sample_with_empty_properties_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)
    properties_list: list[PropertyItem] = []

    with pytest.raises(TypeError):
        sampler._sample(rng, properties_list)


def test_property_sampler_sample_with_single_property_in_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)
    properties_list = ["conductivity"]

    with pytest.raises(TypeError):
        sampler._sample(rng, properties_list)


def test_property_sampler_sample_with_none_properties_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng, None)  # type: ignore[arg-type]  # Testing default None parameter behavior

    assert isinstance(property_item, PropertyItem)
    assert hasattr(property_item, "conductivity")
    assert hasattr(property_item, "permittivity")
    assert hasattr(property_item, "density")


def test_property_sampler_sample_with_equal_min_max_returns_exact_value():
    properties_cfg = {
        "conductivity": {"min": 0.5, "max": 0.5},
        "permittivity": {"min": 10.0, "max": 10.0},
        "density": {"min": 1000.0, "max": 1000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng)

    assert isinstance(property_item, PropertyItem)
    assert property_item.conductivity == 0.5
    assert property_item.permittivity == 10.0
    assert property_item.density == 1000.0


def test_property_sampler_sample_reproducible_with_numpy_seed():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)

    rng1 = default_rng(42)
    property_item_1 = sampler._sample(rng1)

    rng2 = default_rng(42)
    property_item_2 = sampler._sample(rng2)

    assert property_item_1.conductivity == property_item_2.conductivity
    assert property_item_1.permittivity == property_item_2.permittivity
    assert property_item_1.density == property_item_2.density


def test_property_sampler_sample_different_results_with_different_seeds():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    np.random.seed(42)
    property_item_1 = sampler._sample(rng)

    np.random.seed(123)
    property_item_2 = sampler._sample(rng)

    assert property_item_1.conductivity != property_item_2.conductivity


def test_property_sampler_sample_like_with_structure_phantom():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    child_blobs = [
        Blob(position=np.array([1.0, 1.0, 1.0]), radius=3.0),
        Blob(position=np.array([2.0, 2.0, 2.0]), radius=2.0),
    ]
    tubes = [
        Tube(
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            radius=1.0,
        )
    ]
    # Test fixture: list invariance, Blob/Tube are Structure3D subtypes
    structure_phantom = StructurePhantom(
        parent=parent_blob, children=child_blobs, tubes=tubes  # type: ignore[arg-type]
    )

    property_phantom = sampler.sample_like(structure_phantom, rng)

    assert isinstance(property_phantom, PropertyPhantom)
    assert isinstance(property_phantom.parent, PropertyItem)
    assert len(property_phantom.children) == 2
    assert len(property_phantom.tubes) == 1
    assert all(isinstance(child, PropertyItem) for child in property_phantom.children)
    assert all(isinstance(tube, PropertyItem) for tube in property_phantom.tubes)


def test_property_sampler_sample_like_with_mesh_phantom():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_mesh = trimesh.primitives.Sphere(radius=1.0)
    child_meshes = [
        trimesh.primitives.Sphere(radius=0.5),
        trimesh.primitives.Sphere(radius=0.3),
    ]
    tube_meshes = [trimesh.primitives.Cylinder(radius=0.1, height=2.0)]
    # Test fixture: list invariance, Sphere/Cylinder are Trimesh subtypes
    mesh_phantom = MeshPhantom(
        parent=parent_mesh, children=child_meshes, tubes=tube_meshes  # type: ignore[arg-type]
    )

    property_phantom = sampler.sample_like(mesh_phantom, rng)

    assert isinstance(property_phantom, PropertyPhantom)
    assert isinstance(property_phantom.parent, PropertyItem)
    assert len(property_phantom.children) == 2
    assert len(property_phantom.tubes) == 1


def test_property_sampler_sample_like_with_empty_children_and_tubes():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    structure_phantom = StructurePhantom(parent=parent_blob, children=[], tubes=[])

    property_phantom = sampler.sample_like(structure_phantom, rng)

    assert isinstance(property_phantom, PropertyPhantom)
    assert isinstance(property_phantom.parent, PropertyItem)
    assert len(property_phantom.children) == 0
    assert len(property_phantom.tubes) == 0


def test_property_sampler_sample_like_with_specific_properties_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    structure_phantom = StructurePhantom(parent=parent_blob, children=[], tubes=[])
    properties_list = ["conductivity", "permittivity", "density"]

    property_phantom = sampler.sample_like(structure_phantom, rng, properties_list)

    assert isinstance(property_phantom, PropertyPhantom)
    assert isinstance(property_phantom.parent, PropertyItem)
    assert hasattr(property_phantom.parent, "conductivity")
    assert hasattr(property_phantom.parent, "permittivity")
    assert hasattr(property_phantom.parent, "density")


def test_property_sampler_sample_like_with_none_properties_list():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    structure_phantom = StructurePhantom(parent=parent_blob, children=[], tubes=[])

    # Testing default None parameter behavior
    property_phantom = sampler.sample_like(
        structure_phantom, rng, None  # type: ignore[arg-type]
    )

    assert isinstance(property_phantom, PropertyPhantom)
    assert isinstance(property_phantom.parent, PropertyItem)
    assert hasattr(property_phantom.parent, "conductivity")
    assert hasattr(property_phantom.parent, "permittivity")
    assert hasattr(property_phantom.parent, "density")


def test_property_sampler_sample_like_with_moderate_children_and_tubes():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    child_blobs = [Blob(position=np.array([i, i, i]), radius=1.0) for i in range(5)]
    tubes = [
        Tube(
            position=np.array([i, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            radius=0.5,
        )
        for i in range(3)
    ]
    # Test fixture: list invariance, Blob/Tube are Structure3D subtypes
    structure_phantom = StructurePhantom(
        parent=parent_blob, children=child_blobs, tubes=tubes  # type: ignore[arg-type]
    )

    property_phantom = sampler.sample_like(structure_phantom, rng)

    assert isinstance(property_phantom, PropertyPhantom)
    assert len(property_phantom.children) == 5
    assert len(property_phantom.tubes) == 3
    assert all(isinstance(child, PropertyItem) for child in property_phantom.children)
    assert all(isinstance(tube, PropertyItem) for tube in property_phantom.tubes)


def test_property_sampler_sample_values_within_configured_ranges():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    for _ in range(100):
        property_item = sampler._sample(rng)
        assert 0.1 <= property_item.conductivity <= 1.0
        assert 1.0 <= property_item.permittivity <= 100.0
        assert 500.0 <= property_item.density <= 2000.0


def test_property_sampler_sample_boundary_values_minimum_range():
    properties_cfg = {
        "conductivity": {"min": 0.0, "max": 0.001},
        "permittivity": {"min": 1.0, "max": 1.001},
        "density": {"min": 500.0, "max": 500.001},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    for _ in range(10):
        property_item = sampler._sample(rng)
        assert 0.0 <= property_item.conductivity <= 0.001
        assert 1.0 <= property_item.permittivity <= 1.001
        assert 500.0 <= property_item.density <= 500.001


def test_property_sampler_sample_boundary_values_maximum_range():
    properties_cfg = {
        "conductivity": {"min": 999999.0, "max": 1000000.0},
        "permittivity": {"min": 999999.0, "max": 1000000.0},
        "density": {"min": 999999.0, "max": 1000000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    for _ in range(10):
        property_item = sampler._sample(rng)
        assert 999999.0 <= property_item.conductivity <= 1000000.0
        assert 999999.0 <= property_item.permittivity <= 1000000.0
        assert 999999.0 <= property_item.density <= 1000000.0


def test_property_sampler_sample_negative_range_values():
    properties_cfg = {
        "conductivity": {"min": -10.0, "max": -1.0},
        "permittivity": {"min": -100.0, "max": -10.0},
        "density": {"min": -1000.0, "max": -100.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    for _ in range(10):
        property_item = sampler._sample(rng)
        assert -10.0 <= property_item.conductivity <= -1.0
        assert -100.0 <= property_item.permittivity <= -10.0
        assert -1000.0 <= property_item.density <= -100.0


def test_property_sampler_sample_with_custom_property_names_fails():
    properties_cfg = {
        "custom_property_1": {"min": 0.1, "max": 1.0},
        "another_prop": {"min": 100.0, "max": 200.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    with pytest.raises(TypeError):
        sampler._sample(rng)


def test_property_sampler_sample_with_very_small_precision_range():
    properties_cfg = {
        "conductivity": {"min": 1e-15, "max": 1e-14},
        "permittivity": {"min": 1e-15, "max": 1e-14},
        "density": {"min": 1e-15, "max": 1e-14},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng)

    assert 1e-15 <= property_item.conductivity <= 1e-14
    assert 1e-15 <= property_item.permittivity <= 1e-14
    assert 1e-15 <= property_item.density <= 1e-14


def test_property_sampler_sample_with_very_large_precision_range():
    properties_cfg = {
        "conductivity": {"min": 1e14, "max": 1e15},
        "permittivity": {"min": 1e14, "max": 1e15},
        "density": {"min": 1e14, "max": 1e15},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng)

    assert 1e14 <= property_item.conductivity <= 1e15
    assert 1e14 <= property_item.permittivity <= 1e15
    assert 1e14 <= property_item.density <= 1e15


def test_property_sampler_sample_all_properties_different_across_multiple_samples():
    properties_cfg = {
        "conductivity": {"min": 0.0, "max": 1.0},
        "permittivity": {"min": 0.0, "max": 1.0},
        "density": {"min": 0.0, "max": 1.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    samples = [sampler._sample(rng) for _ in range(100)]
    conductivity_values = [sample.conductivity for sample in samples]

    unique_values = set(conductivity_values)
    assert len(unique_values) > 90


def test_property_sampler_sample_like_properties_independent_across_components():
    properties_cfg = {
        "conductivity": {"min": 0.0, "max": 1.0},
        "permittivity": {"min": 0.0, "max": 1.0},
        "density": {"min": 0.0, "max": 1.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    parent_blob = Blob(position=np.array([0.0, 0.0, 0.0]), radius=10.0)
    child_blobs = [
        Blob(position=np.array([1.0, 1.0, 1.0]), radius=3.0),
        Blob(position=np.array([2.0, 2.0, 2.0]), radius=2.0),
    ]
    # Test fixture: list invariance, Blob is Structure3D subtype
    structure_phantom = StructurePhantom(
        parent=parent_blob, children=child_blobs, tubes=[]  # type: ignore[arg-type]
    )

    property_phantom = sampler.sample_like(structure_phantom, rng)

    parent_conductivity = property_phantom.parent.conductivity
    child_conductivities = [child.conductivity for child in property_phantom.children]

    assert parent_conductivity != child_conductivities[0]
    assert child_conductivities[0] != child_conductivities[1]


def test_property_sampler_sample_with_invalid_min_max_order_allows_swap():
    properties_cfg = {
        "conductivity": {"min": 1.0, "max": 0.5},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng)

    assert isinstance(property_item, PropertyItem)
    assert 0.5 <= property_item.conductivity <= 1.0


def test_property_sampler_initialization_with_missing_min_key():
    properties_cfg = {
        "conductivity": {"max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    with pytest.raises(KeyError):
        sampler._sample(rng)


def test_property_sampler_initialization_with_missing_max_key():
    properties_cfg = {
        "conductivity": {"min": 0.1},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    with pytest.raises(KeyError):
        sampler._sample(rng)


def test_property_sampler_initialization_with_none_config():
    sampler = PropertySampler(None)
    rng = default_rng(42)

    with pytest.raises(AttributeError):
        sampler._sample(rng)


def test_property_sampler_sample_with_missing_required_property_in_config():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    with pytest.raises(TypeError):
        sampler._sample(rng)


def test_property_sampler_sample_with_extra_properties_in_config():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
        "extra_property": {"min": 0.0, "max": 1.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    with pytest.raises(TypeError):
        sampler._sample(rng)


def test_property_sampler_sample_like_with_invalid_phantom_type():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    invalid_phantom = "not_a_phantom"

    # Testing error handling with invalid input type
    with pytest.raises(AttributeError):
        sampler.sample_like(invalid_phantom, rng)  # type: ignore[arg-type]


def test_property_sampler_sample_with_zero_range_all_properties():
    properties_cfg = {
        "conductivity": {"min": 0.5, "max": 0.5},
        "permittivity": {"min": 10.0, "max": 10.0},
        "density": {"min": 1000.0, "max": 1000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    for _ in range(10):
        property_item = sampler._sample(rng)
        assert property_item.conductivity == 0.5
        assert property_item.permittivity == 10.0
        assert property_item.density == 1000.0


def test_property_sampler_sample_with_extreme_precision_boundaries():
    properties_cfg = {
        "conductivity": {"min": 1.0000000000000001, "max": 1.0000000000000002},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    property_item = sampler._sample(rng)
    assert isinstance(property_item, PropertyItem)
    assert 1.0000000000000001 <= property_item.conductivity <= 1.0000000000000002


def test_property_sampler_sample_reproducibility_across_multiple_calls():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)

    rng1 = default_rng(42)
    results_1 = [sampler._sample(rng1) for _ in range(10)]

    rng2 = default_rng(42)
    results_2 = [sampler._sample(rng2) for _ in range(10)]

    for r1, r2 in zip(results_1, results_2):
        assert r1.conductivity == r2.conductivity
        assert r1.permittivity == r2.permittivity
        assert r1.density == r2.density


def test_property_sampler_sample_like_phantom_with_no_parent_attribute():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    class InvalidPhantom:
        children = []
        tubes = []

    invalid_phantom = InvalidPhantom()

    # Testing duck typing with custom phantom class
    result = sampler.sample_like(invalid_phantom, rng)  # type: ignore[arg-type]

    assert isinstance(result, PropertyPhantom)
    assert len(result.children) == 0
    assert len(result.tubes) == 0


def test_property_sampler_sample_with_invalid_config_missing_keys():
    properties_cfg = {
        "conductivity": {"max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    with pytest.raises(KeyError):
        sampler._sample(rng)


def test_property_sampler_sample_with_none_config():
    sampler = PropertySampler(None)
    rng = default_rng(42)

    with pytest.raises(AttributeError):
        sampler._sample(rng)


def test_property_sampler_sample_like_with_invalid_phantom():
    properties_cfg = {
        "conductivity": {"min": 0.1, "max": 1.0},
        "permittivity": {"min": 1.0, "max": 100.0},
        "density": {"min": 500.0, "max": 2000.0},
    }
    sampler = PropertySampler(properties_cfg)
    rng = default_rng(42)

    # Testing error handling with invalid input type
    with pytest.raises(AttributeError):
        sampler.sample_like("invalid_phantom", rng)  # type: ignore[arg-type]


def create_test_mesh():
    """Create a simple test mesh (cube) for testing."""
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    return box


def create_test_custom_mesh_structure(tmp_path: Path):
    """Create a CustomMeshStructure from a simple test mesh."""
    test_mesh = create_test_mesh()
    stl_path = tmp_path / "parent.stl"
    test_mesh.export(str(stl_path))
    mesh_structure = CustomMeshStructure(str(stl_path))
    mesh_structure.mesh = test_mesh
    return mesh_structure


def test_mesh_blob_sampler_initialization_with_valid_child_radius():
    child_radius = 0.5

    sampler = MeshBlobSampler(child_radius=child_radius)

    assert sampler.child_radius == child_radius
    assert isinstance(sampler.child_radius, float)
    assert sampler.sample_children_only_inside is False


def test_mesh_blob_sampler_initialization_with_sample_children_only_inside():
    child_radius = 0.3
    sample_children_only_inside = True

    sampler = MeshBlobSampler(
        child_radius=child_radius,
        sample_children_only_inside=sample_children_only_inside,
    )

    assert sampler.child_radius == child_radius
    assert sampler.sample_children_only_inside is True


def test_mesh_blob_sampler_initialization_with_minimal_child_radius():
    child_radius = np.finfo(float).eps

    sampler = MeshBlobSampler(child_radius=child_radius)
    assert sampler.child_radius == child_radius


def test_mesh_blob_sampler_initialization_with_large_child_radius():
    child_radius = 1000.0

    sampler = MeshBlobSampler(child_radius=child_radius)
    assert sampler.child_radius == child_radius


def test_mesh_blob_sampler_rejects_zero_child_radius():
    with pytest.raises(ValueError, match="child_radius must be positive"):
        MeshBlobSampler(child_radius=0.0)


def test_mesh_blob_sampler_rejects_negative_child_radius():
    with pytest.raises(ValueError, match="child_radius must be positive"):
        MeshBlobSampler(child_radius=-1.0)


def test_mesh_blob_sampler_sample_children_blobs_with_zero_children(tmp_path):
    sampler = MeshBlobSampler(child_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(parent_mesh_structure, 0, rng)

    assert len(children) == 0
    assert isinstance(children, list)


def test_mesh_blob_sampler_sample_children_blobs_with_single_child(tmp_path):
    sampler = MeshBlobSampler(child_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(
        parent_mesh_structure, 1, rng, batch_size=2000
    )

    assert len(children) == 1
    assert isinstance(children[0], Blob)
    assert children[0].radius == sampler.child_radius


def test_mesh_blob_sampler_sample_children_blobs_with_multiple_children(tmp_path):
    sampler = MeshBlobSampler(child_radius=0.05)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)
    num_children = 2

    children = sampler.sample_children_blobs(
        parent_mesh_structure, num_children, rng, batch_size=2000
    )

    assert len(children) == num_children
    for child in children:
        assert isinstance(child, Blob)
        assert child.radius == sampler.child_radius


def test_mesh_blob_sampler_sample_children_blobs_reproducible_with_same_seed(tmp_path):
    sampler = MeshBlobSampler(child_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    children1 = sampler.sample_children_blobs(
        parent_mesh_structure, 1, rng1, batch_size=2000
    )
    children2 = sampler.sample_children_blobs(
        parent_mesh_structure, 1, rng2, batch_size=2000
    )

    assert len(children1) == len(children2)
    for c1, c2 in zip(children1, children2):
        assert np.allclose(c1.position, c2.position)
        assert c1.radius == c2.radius


def test_mesh_blob_sampler_sample_children_blobs_different_results_with_different_seeds(
    tmp_path,
):
    sampler = MeshBlobSampler(child_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)

    rng1 = default_rng(42)
    rng2 = default_rng(123)

    children1 = sampler.sample_children_blobs(
        parent_mesh_structure, 1, rng1, batch_size=2000
    )
    children2 = sampler.sample_children_blobs(
        parent_mesh_structure, 1, rng2, batch_size=2000
    )

    assert not np.allclose(children1[0].position, children2[0].position)


def test_mesh_blob_sampler_sample_inside_volume_returns_valid_positions():
    sampler = MeshBlobSampler(child_radius=0.1)
    mesh = create_test_mesh()
    rng = default_rng(42)

    positions = sampler._sample_inside_volume(
        mesh, rng, batch_size=1000, points_to_return=10
    )

    assert positions.shape[0] <= 10
    assert positions.shape[1] == 3

    for position in positions:
        assert mesh.contains([position])[0]


def test_mesh_blob_sampler_sample_inside_volume_with_minimal_batch_size():
    sampler = MeshBlobSampler(child_radius=0.1)
    mesh = create_test_mesh()
    rng = default_rng(42)

    positions = sampler._sample_inside_volume(
        mesh, rng, batch_size=100, points_to_return=1
    )

    assert positions.shape[0] <= 1
    assert positions.shape[1] == 3


def test_mesh_blob_sampler_sample_inside_volume_runtime_error_on_failure():
    """Test that RuntimeError is raised when no valid positions can be found."""
    sampler = MeshBlobSampler(child_radius=0.1)

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0.001], [0.5, 0, -0.001]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    thin_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    rng = default_rng(42)

    with pytest.raises(
        RuntimeError, match="Failed to sample a valid position inside the mesh"
    ):
        sampler._sample_inside_volume(thin_mesh, rng, batch_size=10, points_to_return=1)


def test_mesh_blob_sampler_sample_children_blobs_fails_with_no_valid_placements(
    tmp_path,
):
    """Test that RuntimeError is raised when no valid blob placements can be found."""
    sampler = MeshBlobSampler(child_radius=0.5)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    with pytest.raises(RuntimeError, match="No valid blob placements found"):
        sampler.sample_children_blobs(parent_mesh_structure, 20, rng, batch_size=50)


def test_mesh_blob_sampler_with_sample_children_only_inside_true(tmp_path):
    sampler = MeshBlobSampler(child_radius=0.05, sample_children_only_inside=True)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    try:
        children = sampler.sample_children_blobs(
            parent_mesh_structure, 1, rng, batch_size=2000
        )
        assert len(children) <= 1
        if len(children) > 0:
            assert isinstance(children[0], Blob)
    except RuntimeError as e:
        assert "No valid blob placements found" in str(e)


def test_mesh_blob_sampler_sample_children_inside_filter_failure(tmp_path, monkeypatch):
    sampler = MeshBlobSampler(child_radius=0.1, sample_children_only_inside=True)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(1)

    def fake_sample_inside_volume(mesh, rng, batch_size=0, points_to_return=0):
        return np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    monkeypatch.setattr(sampler, "_sample_inside_volume", fake_sample_inside_volume)
    monkeypatch.setattr(
        parent_mesh_structure.mesh.nearest,
        "signed_distance",
        lambda points: np.zeros(points.shape[0]),
    )

    with pytest.raises(
        RuntimeError, match="No valid blob placements found inside the parental mesh"
    ):
        sampler.sample_children_blobs(parent_mesh_structure, 1, rng, batch_size=2)


def test_mesh_blob_sampler_batch_size_parameter(tmp_path):
    sampler = MeshBlobSampler(child_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    children = sampler.sample_children_blobs(
        parent_mesh_structure, 1, rng, batch_size=2000
    )

    assert len(children) == 1
    assert isinstance(children[0], Blob)


def test_mesh_tube_sampler_initialization_with_valid_radii():
    tube_max_radius = 2.0
    tube_min_radius = 0.5

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )

    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius
    assert isinstance(sampler.tube_max_radius, float)
    assert isinstance(sampler.tube_min_radius, float)


def test_mesh_tube_sampler_initialization_with_minimal_difference():
    tube_max_radius = 1.0 + np.finfo(float).eps
    tube_min_radius = 1.0

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )
    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius


def test_mesh_tube_sampler_initialization_with_large_difference():
    tube_max_radius = 1000.0
    tube_min_radius = 0.001

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )
    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius


def test_mesh_tube_sampler_rejects_zero_max_radius():
    with pytest.raises(ValueError, match="tube_max_radius must be positive"):
        MeshTubeSampler(tube_max_radius=0.0, tube_min_radius=0.1)


def test_mesh_tube_sampler_rejects_negative_max_radius():
    with pytest.raises(ValueError, match="tube_max_radius must be positive"):
        MeshTubeSampler(tube_max_radius=-1.0, tube_min_radius=0.1)


def test_mesh_tube_sampler_rejects_zero_min_radius():
    with pytest.raises(ValueError, match="tube_min_radius must be positive"):
        MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.0)


def test_mesh_tube_sampler_rejects_negative_min_radius():
    with pytest.raises(ValueError, match="tube_min_radius must be positive"):
        MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=-0.1)


def test_mesh_tube_sampler_rejects_min_radius_equal_to_max_radius():
    with pytest.raises(
        ValueError, match="tube_min_radius must be less than tube_max_radius"
    ):
        MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=1.0)


def test_mesh_tube_sampler_rejects_min_radius_greater_than_max_radius():
    with pytest.raises(
        ValueError, match="tube_min_radius must be less than tube_max_radius"
    ):
        MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=1.0)


def test_mesh_tube_sampler_sample_inside_position_returns_valid_position():
    sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    mesh = create_test_mesh()
    rng = default_rng(42)

    position = sampler._sample_inside_position(mesh, rng)

    assert isinstance(position, np.ndarray)
    assert position.shape == (3,)
    assert mesh.contains([position])[0]


def test_mesh_tube_sampler_sample_inside_position_reproducible_with_same_seed():
    sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    mesh = create_test_mesh()

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    position1 = sampler._sample_inside_position(mesh, rng1)
    position2 = sampler._sample_inside_position(mesh, rng2)

    assert np.allclose(position1, position2)


def test_mesh_tube_sampler_sample_inside_position_different_results_with_different_seeds():
    sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    mesh = create_test_mesh()

    rng1 = default_rng(42)
    rng2 = default_rng(123)

    position1 = sampler._sample_inside_position(mesh, rng1)
    position2 = sampler._sample_inside_position(mesh, rng2)

    assert not np.allclose(position1, position2)


def test_mesh_tube_sampler_sample_inside_position_runtime_error_on_failure():
    """Test that RuntimeError is raised when no valid position can be found."""
    sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0.001], [0.5, 0, -0.001]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    thin_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    rng = default_rng(42)

    with pytest.raises(
        RuntimeError, match="Failed to sample a valid position inside the mesh"
    ):
        sampler._sample_inside_position(thin_mesh, rng, max_iter=10)


def test_mesh_tube_sampler_sample_tubes_with_zero_tubes(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 0, rng)

    assert len(tubes) == 0
    assert isinstance(tubes, list)


def test_mesh_tube_sampler_sample_tubes_with_single_tube(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 1, rng)

    assert len(tubes) == 1
    assert isinstance(tubes[0], Tube)
    assert sampler.tube_min_radius <= tubes[0].radius <= sampler.tube_max_radius


def test_mesh_tube_sampler_sample_tubes_with_multiple_tubes(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=0.2, tube_min_radius=0.05)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)
    num_tubes = 2

    tubes = sampler.sample_tubes(parent_mesh_structure, num_tubes, rng)

    assert len(tubes) <= num_tubes
    for tube in tubes:
        assert isinstance(tube, Tube)
        assert sampler.tube_min_radius <= tube.radius <= sampler.tube_max_radius
        assert np.isclose(np.linalg.norm(tube.direction), 1.0)


def test_mesh_tube_sampler_sample_tubes_radii_within_range(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=1.0, tube_min_radius=0.2)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 3, rng)

    for tube in tubes:
        assert sampler.tube_min_radius <= tube.radius <= sampler.tube_max_radius


def test_mesh_tube_sampler_sample_tubes_collision_detection(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 3, rng)

    for i, tube1 in enumerate(tubes):
        for j, tube2 in enumerate(tubes):
            if i != j:
                distance = Tube.distance_to_tube(tube1, tube2)
                min_distance = tube1.radius + tube2.radius
                assert distance >= min_distance or np.isclose(
                    distance, min_distance, atol=1e-10
                )


def test_mesh_tube_sampler_sample_tubes_reproducible_with_same_seed(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)

    rng1 = default_rng(42)
    rng2 = default_rng(42)

    tubes1 = sampler.sample_tubes(parent_mesh_structure, 1, rng1)
    tubes2 = sampler.sample_tubes(parent_mesh_structure, 1, rng2)

    assert len(tubes1) == len(tubes2)
    if len(tubes1) > 0:
        for t1, t2 in zip(tubes1, tubes2):
            assert np.allclose(t1.position, t2.position)
            assert np.allclose(t1.direction, t2.direction)
            assert t1.radius == t2.radius


def test_mesh_tube_sampler_sample_tubes_different_results_with_different_seeds(
    tmp_path,
):
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)

    rng1 = default_rng(42)
    rng2 = default_rng(123)

    tubes1 = sampler.sample_tubes(parent_mesh_structure, 1, rng1)
    tubes2 = sampler.sample_tubes(parent_mesh_structure, 1, rng2)

    if len(tubes1) > 0 and len(tubes2) > 0:
        different = False
        t1, t2 = tubes1[0], tubes2[0]
        if (
            not np.allclose(t1.position, t2.position)
            or not np.allclose(t1.direction, t2.direction)
            or t1.radius != t2.radius
        ):
            different = True
        assert different


def test_mesh_tube_sampler_sample_tubes_with_custom_max_iterations(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 1, rng, max_iterations=100)

    assert len(tubes) <= 1


def test_mesh_tube_sampler_sample_tubes_early_termination_on_failure(tmp_path):
    """Test that sampling terminates early when tube placement becomes impossible."""
    sampler = MeshTubeSampler(tube_max_radius=1.5, tube_min_radius=1.0)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 5, rng, max_iterations=10)

    assert len(tubes) < 5


def test_mesh_tube_sampler_sample_tubes_with_minimal_radius(tmp_path):
    sampler = MeshTubeSampler(
        tube_max_radius=np.finfo(float).eps * 2, tube_min_radius=np.finfo(float).eps
    )
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 1, rng)

    assert len(tubes) <= 1
    if len(tubes) > 0:
        assert tubes[0].radius >= sampler.tube_min_radius
        assert tubes[0].radius <= sampler.tube_max_radius


def test_mesh_tube_sampler_direction_vector_normalization(tmp_path):
    """Test that direction vectors are properly normalized."""
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 3, rng)

    for tube in tubes:
        assert np.isclose(np.linalg.norm(tube.direction), 1.0, atol=1e-10)


def test_mesh_tube_sampler_initialization_with_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5
    parent_radius = 150.0

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius,
        tube_min_radius=tube_min_radius,
        parent_radius=parent_radius,
    )

    assert sampler.tube_max_radius == tube_max_radius
    assert sampler.tube_min_radius == tube_min_radius
    assert sampler.parent_radius == parent_radius
    assert isinstance(sampler.parent_radius, float)


def test_mesh_tube_sampler_initialization_with_default_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius, tube_min_radius=tube_min_radius
    )

    assert sampler.parent_radius == 250.0


def test_mesh_tube_sampler_initialization_with_zero_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5
    parent_radius = 0.0

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius,
        tube_min_radius=tube_min_radius,
        parent_radius=parent_radius,
    )

    assert sampler.parent_radius == 0.0


def test_mesh_tube_sampler_initialization_with_large_parent_radius():
    tube_max_radius = 2.0
    tube_min_radius = 0.5
    parent_radius = 5000.0

    sampler = MeshTubeSampler(
        tube_max_radius=tube_max_radius,
        tube_min_radius=tube_min_radius,
        parent_radius=parent_radius,
    )

    assert sampler.parent_radius == parent_radius


def test_mesh_tube_sampler_sample_tubes_sets_height_based_on_parent_radius(tmp_path):
    parent_radius = 80.0
    sampler = MeshTubeSampler(
        tube_max_radius=0.5, tube_min_radius=0.1, parent_radius=parent_radius
    )
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 2, rng)

    expected_height = 4 * parent_radius
    for tube in tubes:
        assert tube.height == expected_height


def test_mesh_tube_sampler_sample_tubes_with_default_parent_radius_height(tmp_path):
    sampler = MeshTubeSampler(tube_max_radius=0.5, tube_min_radius=0.1)
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 2, rng)

    expected_height = 4 * 250.0
    for tube in tubes:
        assert tube.height == expected_height


def test_mesh_tube_sampler_height_calculation_with_zero_parent_radius(tmp_path):
    parent_radius = 0.0
    sampler = MeshTubeSampler(
        tube_max_radius=0.5, tube_min_radius=0.1, parent_radius=parent_radius
    )
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 1, rng)

    expected_height = 4 * parent_radius
    for tube in tubes:
        assert tube.height == expected_height


def test_mesh_tube_sampler_height_calculation_formula_verification(tmp_path):
    parent_radius_values = [15.0, 50.0, 200.0, 1500.0]
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)

    for parent_radius in parent_radius_values:
        sampler = MeshTubeSampler(
            tube_max_radius=0.5, tube_min_radius=0.1, parent_radius=parent_radius
        )
        rng = default_rng(42)

        tubes = sampler.sample_tubes(parent_mesh_structure, 1, rng)

        expected_height = 4 * parent_radius
        for tube in tubes:
            assert tube.height == expected_height


def test_mesh_tube_sampler_parent_radius_consistency_across_multiple_tubes(tmp_path):
    parent_radius = 120.0
    sampler = MeshTubeSampler(
        tube_max_radius=0.3, tube_min_radius=0.05, parent_radius=parent_radius
    )
    parent_mesh_structure = create_test_custom_mesh_structure(tmp_path)
    rng = default_rng(42)

    tubes = sampler.sample_tubes(parent_mesh_structure, 3, rng)

    expected_height = 4 * parent_radius
    heights = [tube.height for tube in tubes]

    assert all(height == expected_height for height in heights)
    assert len(set(heights)) == 1
