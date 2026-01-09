import pytest
import numpy as np
import os
import tempfile
import trimesh

from magnet_pinn.generator.phantoms import Phantom, Tissue, CustomPhantom
from magnet_pinn.generator.structures import Blob, Tube, CustomMeshStructure
from magnet_pinn.generator.typing import StructurePhantom
from magnet_pinn.generator.samplers import BlobSampler, TubeSampler, MeshBlobSampler, MeshTubeSampler


class ConcretePhantom(Phantom):
    def generate(self, seed=None):
        return StructurePhantom(
            parent=Blob(np.array([0.0, 0.0, 0.0]), 1.0),
            children=[],
            tubes=[]
        )


def test_phantom_initialization_with_valid_parameters():
    initial_blob_radius = 5.0
    initial_blob_center_extent = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])

    phantom = ConcretePhantom(initial_blob_radius, initial_blob_center_extent)

    assert phantom.initial_blob_radius == initial_blob_radius
    assert np.array_equal(phantom.initial_blob_center_extent, initial_blob_center_extent)


def test_phantom_initialization_with_zero_radius():
    initial_blob_radius = 0.0
    initial_blob_center_extent = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    phantom = ConcretePhantom(initial_blob_radius, initial_blob_center_extent)

    assert phantom.initial_blob_radius == 0.0


def test_phantom_initialization_with_large_radius():
    initial_blob_radius = 1000.0
    initial_blob_center_extent = np.array([[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]])

    phantom = ConcretePhantom(initial_blob_radius, initial_blob_center_extent)

    assert phantom.initial_blob_radius == 1000.0


def test_phantom_initialization_with_negative_extent():
    initial_blob_radius = 1.0
    initial_blob_center_extent = np.array([[-10.0, -5.0], [-10.0, -5.0], [-10.0, -5.0]])

    phantom = ConcretePhantom(initial_blob_radius, initial_blob_center_extent)

    assert np.array_equal(phantom.initial_blob_center_extent, initial_blob_center_extent)


def test_phantom_initialization_with_single_point_extent():
    initial_blob_radius = 1.0
    initial_blob_center_extent = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    phantom = ConcretePhantom(initial_blob_radius, initial_blob_center_extent)

    assert np.array_equal(phantom.initial_blob_center_extent, initial_blob_center_extent)


def test_phantom_generate_raises_not_implemented_error():
    phantom = Phantom(1.0, np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

    with pytest.raises(NotImplementedError, match="Subclasses should implement this method"):
        phantom.generate()


def test_tissue_initialization_with_valid_parameters():
    num_children_blobs = 3
    initial_blob_radius = 10.0
    initial_blob_center_extent = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    blob_radius_decrease_per_level = 0.5
    num_tubes = 2
    relative_tube_max_radius = 0.1
    relative_tube_min_radius = 0.05

    tissue = Tissue(
        num_children_blobs,
        initial_blob_radius,
        initial_blob_center_extent,
        blob_radius_decrease_per_level,
        num_tubes,
        relative_tube_max_radius,
        relative_tube_min_radius
    )

    assert tissue.num_children_blobs == num_children_blobs
    assert tissue.num_tubes == num_tubes
    assert isinstance(tissue.blob_sampler, BlobSampler)
    assert isinstance(tissue.tube_sampler, TubeSampler)


def test_tissue_initialization_with_zero_children():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=5.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.3,
        num_tubes=1,
        relative_tube_max_radius=0.2
    )

    assert tissue.num_children_blobs == 0


def test_tissue_initialization_with_zero_tubes():
    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=5.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.3,
        num_tubes=0,
        relative_tube_max_radius=0.2
    )

    assert tissue.num_tubes == 0


def test_tissue_initialization_with_minimum_blob_radius_decrease():
    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=1.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.001,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    assert tissue.blob_sampler.radius_decrease_factor == 0.001


def test_tissue_initialization_with_maximum_blob_radius_decrease():
    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=1.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.999,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    assert tissue.blob_sampler.radius_decrease_factor == 0.999


def test_tissue_initialization_with_minimum_tube_radii():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=1,
        relative_tube_max_radius=0.001,
        relative_tube_min_radius=0.0001
    )

    assert tissue.tube_sampler.tube_max_radius == 0.01
    assert tissue.tube_sampler.tube_min_radius == 0.001


def test_tissue_initialization_with_maximum_tube_radii():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=1,
        relative_tube_max_radius=0.999,
        relative_tube_min_radius=0.1
    )

    assert tissue.tube_sampler.tube_max_radius == 9.99
    assert tissue.tube_sampler.tube_min_radius == 1.0


def test_tissue_initialization_with_default_min_tube_radius():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=1,
        relative_tube_max_radius=0.2
    )

    assert tissue.tube_sampler.tube_min_radius == 0.1


def test_tissue_rejects_negative_children_blobs():
    with pytest.raises(ValueError, match="num_children_blobs must be non-negative"):
        Tissue(
            num_children_blobs=-1,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_zero_initial_blob_radius():
    with pytest.raises(ValueError, match="initial_blob_radius must be positive"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=0.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_negative_initial_blob_radius():
    with pytest.raises(ValueError, match="initial_blob_radius must be positive"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=-1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_negative_num_tubes():
    with pytest.raises(ValueError, match="num_tubes must be non-negative"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=-1,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_zero_relative_tube_max_radius():
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.0
        )


def test_tissue_rejects_relative_tube_max_radius_equal_to_one():
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=1.0
        )


def test_tissue_rejects_relative_tube_max_radius_greater_than_one():
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=1.5
        )


def test_tissue_rejects_zero_relative_tube_min_radius():
    with pytest.raises(ValueError, match="relative_tube_min_radius must be in \\(0, relative_tube_max_radius\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.2,
            relative_tube_min_radius=0.0
        )


def test_tissue_rejects_negative_relative_tube_min_radius():
    with pytest.raises(ValueError, match="relative_tube_min_radius must be in \\(0, relative_tube_max_radius\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.2,
            relative_tube_min_radius=-0.1
        )


def test_tissue_rejects_relative_tube_min_radius_equal_to_max():
    with pytest.raises(ValueError, match="relative_tube_min_radius must be in \\(0, relative_tube_max_radius\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.2,
            relative_tube_min_radius=0.2
        )


def test_tissue_rejects_relative_tube_min_radius_greater_than_max():
    with pytest.raises(ValueError, match="relative_tube_min_radius must be in \\(0, relative_tube_max_radius\\)"):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.2,
            relative_tube_min_radius=0.3
        )


def test_tissue_rejects_none_initial_blob_center_extent():
    match_msg = (
        "initial_blob_center_extent must be a 2d array-like structure with coordinate ranges, "
        "first X, then Y, then Z dimensions"
    )
    # Testing None validation for initial_blob_center_extent
    with pytest.raises(ValueError, match=match_msg):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=None,  # type: ignore[arg-type]
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.2
        )


def test_tissue_rejects_list_of_lists_initial_blob_center_extent():
    match_msg = (
        "initial_blob_center_extent must be a 2d array-like structure with coordinate ranges, "
        "first X, then Y, then Z dimensions"
    )
    # Testing list-to-array coercion validation
    with pytest.raises(ValueError, match=match_msg):
        Tissue(
            num_children_blobs=0,
            initial_blob_radius=1.0,
            initial_blob_center_extent=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # type: ignore[arg-type]
            blob_radius_decrease_per_level=0.5,
            num_tubes=0,
            relative_tube_max_radius=0.2
        )


def test_tissue_generate_with_zero_children_and_tubes():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=5.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=0,
        relative_tube_max_radius=0.2
    )

    phantom = tissue.generate(seed=42)

    assert isinstance(phantom, StructurePhantom)
    assert isinstance(phantom.parent, Blob)
    assert len(phantom.children) == 0
    assert len(phantom.tubes) == 0


def test_tissue_generate_with_children_only():
    tissue = Tissue(
        num_children_blobs=2,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        blob_radius_decrease_per_level=0.3,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    assert isinstance(phantom, StructurePhantom)
    assert isinstance(phantom.parent, Blob)
    assert len(phantom.children) == 2
    assert len(phantom.tubes) == 0
    assert all(isinstance(child, Blob) for child in phantom.children)


def test_tissue_generate_with_tubes_only():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        blob_radius_decrease_per_level=0.3,
        num_tubes=2,
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    assert isinstance(phantom, StructurePhantom)
    assert isinstance(phantom.parent, Blob)
    assert len(phantom.children) == 0
    assert len(phantom.tubes) == 2
    assert all(isinstance(tube, Tube) for tube in phantom.tubes)


def test_tissue_generate_with_children_and_tubes():
    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]),
        blob_radius_decrease_per_level=0.4,
        num_tubes=1,
        relative_tube_max_radius=0.15
    )

    phantom = tissue.generate(seed=42)

    assert isinstance(phantom, StructurePhantom)
    assert isinstance(phantom.parent, Blob)
    assert len(phantom.children) == 1
    assert len(phantom.tubes) == 1
    assert isinstance(phantom.children[0], Blob)
    assert isinstance(phantom.tubes[0], Tube)


def test_tissue_generate_parent_blob_within_extent():
    extent = np.array([[-5.0, 5.0], [-3.0, 3.0], [-1.0, 1.0]])
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=2.0,
        initial_blob_center_extent=extent,
        blob_radius_decrease_per_level=0.5,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    parent_pos = phantom.parent.position
    assert extent[0, 0] <= parent_pos[0] <= extent[0, 1]
    assert extent[1, 0] <= parent_pos[1] <= extent[1, 1]
    assert extent[2, 0] <= parent_pos[2] <= extent[2, 1]


def test_tissue_generate_parent_blob_correct_radius():
    initial_blob_radius = 7.5
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=initial_blob_radius,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    assert phantom.parent.radius == initial_blob_radius


def test_tissue_generate_child_blobs_correct_radius():
    initial_blob_radius = 10.0
    radius_decrease = 0.3
    tissue = Tissue(
        num_children_blobs=2,
        initial_blob_radius=initial_blob_radius,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=radius_decrease,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    expected_child_radius = initial_blob_radius * radius_decrease
    for child in phantom.children:
        assert child.radius == expected_child_radius


def test_tissue_generate_tube_radii_within_range():
    initial_blob_radius = 10.0
    max_relative = 0.2
    min_relative = 0.1
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=initial_blob_radius,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=3,
        relative_tube_max_radius=max_relative,
        relative_tube_min_radius=min_relative
    )

    phantom = tissue.generate(seed=42)

    expected_min = min_relative * initial_blob_radius
    expected_max = max_relative * initial_blob_radius
    for tube in phantom.tubes:
        assert expected_min <= tube.radius <= expected_max


def test_tissue_generate_reproducible_with_same_seed():
    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=5.0,
        initial_blob_center_extent=np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        blob_radius_decrease_per_level=0.4,
        num_tubes=1,
        relative_tube_max_radius=0.2
    )

    phantom1 = tissue.generate(seed=42)
    phantom2 = tissue.generate(seed=42)

    assert np.allclose(phantom1.parent.position, phantom2.parent.position)
    assert phantom1.parent.radius == phantom2.parent.radius
    assert len(phantom1.children) == len(phantom2.children)
    assert len(phantom1.tubes) == len(phantom2.tubes)


def test_tissue_generate_different_results_with_different_seeds():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=5.0,
        initial_blob_center_extent=np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]),
        blob_radius_decrease_per_level=0.4,
        num_tubes=0,
        relative_tube_max_radius=0.2
    )

    phantom1 = tissue.generate(seed=42)
    phantom2 = tissue.generate(seed=123)

    assert not np.allclose(phantom1.parent.position, phantom2.parent.position)


def test_tissue_generate_without_seed():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=5.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.4,
        num_tubes=0,
        relative_tube_max_radius=0.2
    )

    phantom = tissue.generate()

    assert isinstance(phantom, StructurePhantom)
    assert isinstance(phantom.parent, Blob)


def test_tissue_generate_tube_direction_vectors_normalized():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=3,
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    # Runtime type of phantom.tubes items is Tube, which has direction attribute
    for tube in phantom.tubes:
        norm = np.linalg.norm(tube.direction)  # type: ignore[attr-defined]
        assert np.isclose(norm, 1.0, rtol=1e-10)


def test_tissue_generate_uses_parent_inner_radius_for_tubes():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=1,
        relative_tube_max_radius=0.1
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
        relative_tube_max_radius=0.1
    )

    phantom = tissue.generate(seed=42)

    assert len(phantom.children) == 1
    child = phantom.children[0]
    # Runtime types are Blob, which have empirical_min_offset and empirical_max_offset
    parent = phantom.parent

    distance = np.linalg.norm(child.position - parent.position)
    max_distance = (
        parent.radius * (1 + parent.empirical_min_offset)  # type: ignore[attr-defined]
        - child.radius * (1 + child.empirical_max_offset)  # type: ignore[attr-defined]
    )
    assert distance <= max_distance


def test_tissue_generate_with_single_tube():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=10.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=1,
        relative_tube_max_radius=0.15
    )

    phantom = tissue.generate(seed=42)

    assert len(phantom.tubes) == 1
    tube = phantom.tubes[0]
    assert isinstance(tube, Tube)
    assert hasattr(tube, 'position')
    assert hasattr(tube, 'direction')
    assert hasattr(tube, 'radius')


def test_tissue_has_correct_sampler_configuration():
    blob_decrease = 0.3
    max_tube_radius = 0.2
    min_tube_radius = 0.05
    initial_radius = 10.0

    tissue = Tissue(
        num_children_blobs=1,
        initial_blob_radius=initial_radius,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=blob_decrease,
        num_tubes=1,
        relative_tube_max_radius=max_tube_radius,
        relative_tube_min_radius=min_tube_radius
    )

    assert tissue.blob_sampler.radius_decrease_factor == blob_decrease
    assert tissue.tube_sampler.tube_max_radius == max_tube_radius * initial_radius
    assert tissue.tube_sampler.tube_min_radius == min_tube_radius * initial_radius


def test_tissue_inheritance_from_phantom():
    tissue = Tissue(
        num_children_blobs=0,
        initial_blob_radius=1.0,
        initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        blob_radius_decrease_per_level=0.5,
        num_tubes=0,
        relative_tube_max_radius=0.1
    )

    assert isinstance(tissue, Phantom)
    assert hasattr(tissue, 'initial_blob_radius')
    assert hasattr(tissue, 'initial_blob_center_extent')


def test_tissue_rejects_zero_blob_radius_decrease_per_level():
    with pytest.raises(ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=1,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=0.0,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_negative_blob_radius_decrease_per_level():
    with pytest.raises(ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=1,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=-0.5,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_blob_radius_decrease_per_level_equal_to_one():
    with pytest.raises(ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=1,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=1.0,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


def test_tissue_rejects_blob_radius_decrease_per_level_greater_than_one():
    with pytest.raises(ValueError, match="radius_decrease_factor must be in \\(0, 1\\)"):
        Tissue(
            num_children_blobs=1,
            initial_blob_radius=1.0,
            initial_blob_center_extent=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            blob_radius_decrease_per_level=1.5,
            num_tubes=0,
            relative_tube_max_radius=0.1
        )


@pytest.fixture(scope='module')
def simple_stl_file(tmp_path_factory):
    """Create a module-scoped STL file for testing purposes.

    Creates an icosphere mesh and exports it to a temporary STL file that
    persists for the entire test module. This avoids repeated file I/O
    for tests that require an STL file.
    """
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2.0)

    stl_dir = tmp_path_factory.mktemp('stl_files')
    stl_path = stl_dir / 'simple.stl'
    mesh.export(str(stl_path))

    yield str(stl_path)


def test_custom_phantom_initialization_with_valid_stl_file(simple_stl_file):
    """Test CustomPhantom initialization with valid STL file."""
    num_children_blobs = 3
    blob_radius_decrease_per_level = 0.3
    num_tubes = 5
    relative_tube_max_radius = 0.1
    relative_tube_min_radius = 0.01
    sample_children_only_inside = False

    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=num_children_blobs,
        blob_radius_decrease_per_level=blob_radius_decrease_per_level,
        num_tubes=num_tubes,
        relative_tube_max_radius=relative_tube_max_radius,
        relative_tube_min_radius=relative_tube_min_radius,
        sample_children_only_inside=sample_children_only_inside
    )

    assert phantom.num_children_blobs == num_children_blobs
    assert phantom.num_tubes == num_tubes
    assert isinstance(phantom.parent_structure, CustomMeshStructure)
    assert isinstance(phantom.child_sampler, MeshBlobSampler)
    assert isinstance(phantom.tube_sampler, MeshTubeSampler)
    assert phantom.initial_blob_radius is None
    assert phantom.initial_blob_center_extent is None


def test_custom_phantom_initialization_with_default_parameters(simple_stl_file):
    """Test CustomPhantom initialization with default parameters."""
    phantom = CustomPhantom(stl_mesh_path=simple_stl_file)

    assert phantom.num_children_blobs == 3
    assert phantom.num_tubes == 5
    assert isinstance(phantom.parent_structure, CustomMeshStructure)
    assert isinstance(phantom.child_sampler, MeshBlobSampler)
    assert isinstance(phantom.tube_sampler, MeshTubeSampler)


def test_custom_phantom_initialization_with_zero_children(simple_stl_file):
    """Test CustomPhantom initialization with zero children blobs."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=0,
        num_tubes=2
    )

    assert phantom.num_children_blobs == 0
    assert phantom.num_tubes == 2


def test_custom_phantom_initialization_with_zero_tubes(simple_stl_file):
    """Test CustomPhantom initialization with zero tubes."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=2,
        num_tubes=0
    )

    assert phantom.num_children_blobs == 2
    assert phantom.num_tubes == 0


def test_custom_phantom_initialization_with_minimum_blob_radius_decrease(simple_stl_file):
    """Test CustomPhantom initialization with minimum blob radius decrease factor."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        blob_radius_decrease_per_level=0.001
    )

    assert phantom.num_children_blobs == 3


def test_custom_phantom_initialization_with_maximum_blob_radius_decrease(simple_stl_file):
    """Test CustomPhantom initialization with maximum blob radius decrease factor."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        blob_radius_decrease_per_level=0.999
    )

    assert phantom.num_children_blobs == 3


def test_custom_phantom_initialization_with_sample_children_only_inside_true(simple_stl_file):
    """Test CustomPhantom initialization with sample_children_only_inside=True."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        sample_children_only_inside=True
    )

    assert phantom.num_children_blobs == 3


def test_custom_phantom_initialization_with_large_tube_radii(simple_stl_file):
    """Test CustomPhantom initialization with large tube radii."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        relative_tube_max_radius=0.9,
        relative_tube_min_radius=0.1
    )

    assert phantom.num_children_blobs == 3
    assert phantom.num_tubes == 5


def test_custom_phantom_initialization_with_small_tube_radii(simple_stl_file):
    """Test CustomPhantom initialization with small tube radii."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        relative_tube_max_radius=0.001,
        relative_tube_min_radius=0.0001
    )

    assert phantom.num_children_blobs == 3
    assert phantom.num_tubes == 5


def test_custom_phantom_initialization_with_nonexistent_stl_file():
    """Test CustomPhantom initialization with non-existent STL file."""
    with pytest.raises((FileNotFoundError, IOError, ValueError)):
        CustomPhantom(stl_mesh_path="/nonexistent/path/phantom.stl")


def test_custom_phantom_initialization_with_invalid_stl_file():
    """Test CustomPhantom initialization with invalid STL file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False)
    temp_file.write("This is not a valid STL file")
    temp_file.close()

    try:
        with pytest.raises((ValueError, Exception)):
            CustomPhantom(stl_mesh_path=temp_file.name)
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def test_custom_phantom_generate_with_zero_children_and_tubes(simple_stl_file):
    """Test CustomPhantom generation with zero children and tubes."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=0,
        num_tubes=0
    )

    result = phantom.generate(seed=42)

    assert isinstance(result, StructurePhantom)
    assert isinstance(result.parent, CustomMeshStructure)
    assert len(result.children) == 0
    assert len(result.tubes) == 0


def test_custom_phantom_generate_with_children_only(simple_stl_file):
    """Test CustomPhantom generation with children only."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=2,
        num_tubes=0
    )

    result = phantom.generate(seed=42)

    assert isinstance(result, StructurePhantom)
    assert isinstance(result.parent, CustomMeshStructure)
    assert len(result.children) == 2
    assert len(result.tubes) == 0

    for child in result.children:
        assert isinstance(child, Blob)


def test_custom_phantom_generate_with_tubes_only(simple_stl_file):
    """Test CustomPhantom generation with tubes only."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=0,
        num_tubes=2
    )

    result = phantom.generate(seed=42)

    assert isinstance(result, StructurePhantom)
    assert isinstance(result.parent, CustomMeshStructure)
    assert len(result.children) == 0
    assert len(result.tubes) == 2

    for tube in result.tubes:
        assert isinstance(tube, Tube)


def test_custom_phantom_generate_with_children_and_tubes(simple_stl_file):
    """Test CustomPhantom generation with both children and tubes."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=2,
        num_tubes=3
    )

    result = phantom.generate(seed=42)

    assert isinstance(result, StructurePhantom)
    assert isinstance(result.parent, CustomMeshStructure)
    assert len(result.children) == 2
    assert len(result.tubes) == 3

    for child in result.children:
        assert isinstance(child, Blob)

    for tube in result.tubes:
        assert isinstance(tube, Tube)


def test_custom_phantom_generate_reproducible_with_same_seed(simple_stl_file):
    """Test CustomPhantom generation is reproducible with same seed."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=2,
        num_tubes=2
    )

    result1 = phantom.generate(seed=42)
    result2 = phantom.generate(seed=42)

    assert len(result1.children) == len(result2.children)
    assert len(result1.tubes) == len(result2.tubes)

    for child1, child2 in zip(result1.children, result2.children):
        np.testing.assert_array_almost_equal(child1.position, child2.position)
        assert child1.radius == child2.radius

    # Runtime type of tubes items is Tube, which has direction attribute
    for tube1, tube2 in zip(result1.tubes, result2.tubes):
        np.testing.assert_array_almost_equal(tube1.position, tube2.position)
        np.testing.assert_array_almost_equal(tube1.direction, tube2.direction)  # type: ignore[attr-defined]
        assert tube1.radius == tube2.radius


def test_custom_phantom_generate_different_results_with_different_seeds(simple_stl_file):
    """Test CustomPhantom generation produces different results with different seeds."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=2,
        num_tubes=2
    )

    result1 = phantom.generate(seed=42)
    result2 = phantom.generate(seed=123)

    children_different = False
    for child1, child2 in zip(result1.children, result2.children):
        if not np.allclose(child1.position, child2.position) or child1.radius != child2.radius:
            children_different = True
            break

    # Runtime type of tubes items is Tube, which has direction attribute
    tubes_different = False
    for tube1, tube2 in zip(result1.tubes, result2.tubes):
        if (
            not np.allclose(tube1.position, tube2.position)
            or not np.allclose(tube1.direction, tube2.direction)  # type: ignore[attr-defined]
            or tube1.radius != tube2.radius
        ):
            tubes_different = True
            break

    assert children_different or tubes_different


def test_custom_phantom_generate_without_seed(simple_stl_file):
    """Test CustomPhantom generation without specifying seed."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=1,
        num_tubes=1
    )

    result = phantom.generate()

    assert isinstance(result, StructurePhantom)
    assert isinstance(result.parent, CustomMeshStructure)
    assert len(result.children) == 1
    assert len(result.tubes) == 1


def test_custom_phantom_generate_with_custom_batch_size(simple_stl_file):
    """Test CustomPhantom generation with custom batch size."""
    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        num_children_blobs=2,
        num_tubes=1
    )

    result = phantom.generate(seed=42, child_blobs_batch_size=500000)

    assert isinstance(result, StructurePhantom)
    assert len(result.children) == 2
    assert len(result.tubes) == 1


def test_custom_phantom_inheritance_from_phantom(simple_stl_file):
    """Test CustomPhantom properly inherits from Phantom base class."""
    phantom = CustomPhantom(stl_mesh_path=simple_stl_file)

    assert isinstance(phantom, Phantom)
    assert hasattr(phantom, 'generate')
    assert callable(phantom.generate)


def test_custom_phantom_sampler_configuration(simple_stl_file):
    """Test CustomPhantom has correct sampler configuration."""
    blob_radius_decrease = 0.4
    tube_max_radius = 0.15
    tube_min_radius = 0.05
    sample_inside = True

    phantom = CustomPhantom(
        stl_mesh_path=simple_stl_file,
        blob_radius_decrease_per_level=blob_radius_decrease,
        relative_tube_max_radius=tube_max_radius,
        relative_tube_min_radius=tube_min_radius,
        sample_children_only_inside=sample_inside
    )

    assert isinstance(phantom.child_sampler, MeshBlobSampler)
    assert isinstance(phantom.tube_sampler, MeshTubeSampler)

    assert isinstance(phantom.parent_structure, CustomMeshStructure)

    expected_child_radius = phantom.parent_structure.radius * blob_radius_decrease
    expected_tube_max = tube_max_radius * phantom.parent_structure.radius
    expected_tube_min = tube_min_radius * phantom.parent_structure.radius

    assert phantom.child_sampler.child_radius == expected_child_radius
    assert phantom.tube_sampler.tube_max_radius == expected_tube_max
    assert phantom.tube_sampler.tube_min_radius == expected_tube_min
