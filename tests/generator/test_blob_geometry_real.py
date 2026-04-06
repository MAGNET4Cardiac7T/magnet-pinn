"""Geometry-sensitive Blob tests that must use the real ``Blob.__init__``.

These tests intentionally live outside ``tests/generation/`` because
``tests/generation/conftest.py`` applies the module-scoped autouse
``fast_blob_initialization`` fixture there, which patches ``Blob.__init__``
with stub empirical offsets. The assertions below rely on real empirical
geometry margins, so they belong in ``tests/generator/``.
"""

import numpy as np
from numpy.random import default_rng

from magnet_pinn.generator.phantoms import Tissue
from magnet_pinn.generator.samplers import BlobSampler
from magnet_pinn.generator.structures import Blob


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


def test_tissue_generate_uses_parent_inner_radius_for_tubes():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=1,
        relative_tube_max_radius=0.1,
    )

    phantom = tissue.generate(seed=42)

    # Runtime type of phantom.parent is Blob, which has empirical_min_offset
    parent = phantom.parent
    expected_max_distance = parent.radius * (1 + parent.empirical_min_offset)  # type: ignore[attr-defined]

    for tube in phantom.tubes:
        distance_to_center = np.linalg.norm(tube.position - parent.position)
        assert distance_to_center + tube.radius <= expected_max_distance


def test_tissue_generate_with_single_child_blob():
    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.2,
        num_tubes=0,
        relative_tube_max_radius=0.1,
    )

    phantom = tissue.generate(seed=42)

    assert len(phantom.children) == 1
    child = phantom.children[0]
    # Runtime types are Blob, which have empirical_min_offset and empirical_max_offset
    parent = phantom.parent

    distance = np.linalg.norm(child.position - parent.position)
    max_distance = parent.radius * (
        1 + parent.empirical_min_offset
    ) - child.radius * (  # type: ignore[attr-defined]
        1 + child.empirical_max_offset
    )  # type: ignore[attr-defined]
    assert distance <= max_distance
