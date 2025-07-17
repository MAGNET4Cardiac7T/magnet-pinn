import pytest
import numpy as np

from magnet_pinn.generator.phantoms import Phantom, Tissue
from magnet_pinn.generator.structures import Blob, Tube
from magnet_pinn.generator.typing import StructurePhantom
from magnet_pinn.generator.samplers import BlobSampler, TubeSampler


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
    
    for tube in phantom.tubes:
        norm = np.linalg.norm(tube.direction)
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
    
    parent = phantom.parent
    expected_max_distance = parent.radius * (1 + parent.empirical_min_offset)
    
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
    parent = phantom.parent
    
    distance = np.linalg.norm(child.position - parent.position)
    max_distance = parent.radius * (1 + parent.empirical_min_offset) - child.radius * (1 + child.empirical_max_offset)
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
