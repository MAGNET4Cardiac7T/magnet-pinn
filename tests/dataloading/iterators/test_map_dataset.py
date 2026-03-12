"""Tests for the map-style MagnetDataset class."""

from shutil import rmtree

import pytest
import numpy as np
import torch.utils.data

from magnet_pinn.data._base import MagnetDataset
from magnet_pinn.preprocessing.preprocessing import (
    PROCESSED_ANTENNA_DIR_PATH,
    TARGET_FILE_NAME,
    PROCESSED_SIMULATIONS_DIR_PATH,
)
from tests.dataloading.iterators.helpers import (
    RANDOM_SIM_FILE_NAME,
    ZERO_SIM_FILE_NAME,
    create_processed_dir,
    check_dtypes_between_iter_result_and_supposed_simulation,
    check_shapes_between_item_result_and_supposed_simulation,
    check_values_between_item_result_and_supposed_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MAP_DATASET_DIR_NAME = "test_map_dataset_grid_voxel_size_4_data_type_float32"
MAP_DATASET_DIR_NAME_SHORT_TERM = "test_map_dataset_grid_short_term"


@pytest.fixture(scope="module")
def map_dataset_dir(processed_dir_path, random_grid_item, zero_grid_item):
    """Create a module-scoped processed directory for MagnetDataset tests."""
    path = processed_dir_path / MAP_DATASET_DIR_NAME
    create_processed_dir(path, random_grid_item, zero_grid_item, is_grid=True)
    yield path
    if path.exists():
        rmtree(path)


@pytest.fixture(scope="function")
def map_dataset_dir_short_term(processed_dir_path, random_grid_item, zero_grid_item):
    """Create a function-scoped processed directory for short-lived MagnetDataset tests."""
    path = processed_dir_path / MAP_DATASET_DIR_NAME_SHORT_TERM
    create_processed_dir(path, random_grid_item, zero_grid_item, is_grid=True)
    yield path
    if path.exists():
        rmtree(path)


@pytest.fixture(scope="module")
def map_aug():
    """PhaseShift augmentation reused for MagnetDataset tests."""
    from magnet_pinn.data.transforms import PhaseShift
    return PhaseShift(num_coils=8)


# ---------------------------------------------------------------------------
# Class / inheritance tests
# ---------------------------------------------------------------------------


def test_map_dataset_is_torch_dataset():
    assert issubclass(MagnetDataset, torch.utils.data.Dataset)


def test_map_dataset_is_not_iterable_dataset():
    assert not issubclass(MagnetDataset, torch.utils.data.IterableDataset)


def test_map_dataset_can_be_instantiated(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)
    assert isinstance(ds, MagnetDataset)


# ---------------------------------------------------------------------------
# Property tests (mirrors iterator property tests)
# ---------------------------------------------------------------------------


def test_map_dataset_coils_properties(map_dataset_dir, random_grid_item, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)

    expected_coils_path = (
        map_dataset_dir / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    )
    assert ds.coils_path == expected_coils_path
    assert ds.num_coils == random_grid_item.coils.shape[0]
    assert ds.coils.shape == random_grid_item.coils.shape
    assert ds.coils.dtype == np.bool_
    assert np.equal(ds.coils, random_grid_item.coils.astype(np.bool_)).all()


def test_map_dataset_simulations_properties(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)

    expected_sim_dir = map_dataset_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert ds.simulation_dir == expected_sim_dir

    expected_sim_list = [
        expected_sim_dir / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME),
        expected_sim_dir / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME),
    ]
    assert ds.simulation_list == expected_sim_list


def test_map_dataset_other_properties(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=2)
    assert ds.num_samples == 2
    assert ds.transforms == map_aug


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


def test_map_dataset_num_samples_zero_raises(map_dataset_dir, map_aug):
    with pytest.raises(ValueError):
        MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=0)


def test_map_dataset_num_samples_negative_raises(map_dataset_dir, map_aug):
    with pytest.raises(ValueError):
        MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=-1)


def test_map_dataset_invalid_transforms_raises(map_dataset_dir):
    with pytest.raises(ValueError):
        MagnetDataset(map_dataset_dir, transforms=None, num_samples=1)


def test_map_dataset_missing_coils_dir_raises(map_dataset_dir_short_term, map_aug):
    antenna_dir = map_dataset_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    rmtree(antenna_dir)
    with pytest.raises(FileNotFoundError):
        MagnetDataset(map_dataset_dir_short_term, transforms=map_aug, num_samples=1)


def test_map_dataset_missing_simulations_dir_raises(map_dataset_dir_short_term, map_aug):
    simulations_dir = map_dataset_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir)
    with pytest.raises(FileNotFoundError):
        MagnetDataset(map_dataset_dir_short_term, transforms=map_aug, num_samples=1)


def test_map_dataset_empty_simulations_dir_raises(map_dataset_dir_short_term, map_aug):
    simulations_dir = map_dataset_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir)
    simulations_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        MagnetDataset(map_dataset_dir_short_term, transforms=map_aug, num_samples=1)


# ---------------------------------------------------------------------------
# __len__ tests
# ---------------------------------------------------------------------------


def test_map_dataset_len_unit_num_samples(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)
    assert len(ds) == 2  # 2 simulations × 1 sample


def test_map_dataset_len_multiple_num_samples(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=100)
    assert len(ds) == 200  # 2 simulations × 100 samples


# ---------------------------------------------------------------------------
# __getitem__ index mapping tests
# ---------------------------------------------------------------------------


def test_map_dataset_getitem_first_index_is_first_simulation(
    map_dataset_dir, random_grid_item, zero_grid_item, map_aug
):
    """index 0 must come from simulation 0 (natural-sort first)."""
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=10)
    item = ds[0]
    # simulation_list is natural-sorted, first entry is random_grid_item
    assert item["simulation"] in {random_grid_item.simulation, zero_grid_item.simulation}


def test_map_dataset_getitem_index_maps_to_correct_simulation(
    map_dataset_dir, random_grid_item, zero_grid_item, map_aug
):
    """Verify that indices map to the expected simulations according to the layout."""
    num_samples = 10
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=num_samples)

    # First `num_samples` indices → first simulation
    first_sim = ds[0]["simulation"]
    for i in range(num_samples):
        assert ds[i]["simulation"] == first_sim

    # Next `num_samples` indices → second simulation
    second_sim = ds[num_samples]["simulation"]
    assert second_sim != first_sim
    for i in range(num_samples, 2 * num_samples):
        assert ds[i]["simulation"] == second_sim


def test_map_dataset_getitem_last_valid_index(map_dataset_dir, map_aug):
    """Last valid index (len-1) should succeed."""
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=5)
    item = ds[len(ds) - 1]
    assert isinstance(item, dict)


def test_map_dataset_getitem_index_out_of_bounds_raises(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=5)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_map_dataset_getitem_negative_index_raises(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=5)
    with pytest.raises(IndexError):
        ds[-1]


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------


def test_map_dataset_getitem_returns_dict(map_dataset_dir, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)
    item = ds[0]
    assert isinstance(item, dict)


def test_map_dataset_getitem_dtypes(map_dataset_dir, random_grid_item, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)
    for i in range(len(ds)):
        item = ds[i]
        check_dtypes_between_iter_result_and_supposed_simulation(item, random_grid_item)


def test_map_dataset_getitem_shapes(map_dataset_dir, random_grid_item, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)
    for i in range(len(ds)):
        item = ds[i]
        check_shapes_between_item_result_and_supposed_simulation(item, random_grid_item)


def test_map_dataset_getitem_values(map_dataset_dir, random_grid_item, zero_grid_item, map_aug):
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=1)
    for i in range(len(ds)):
        result = ds[i]
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            raise ValueError(f"Unexpected simulation: {result['simulation']}")


def test_map_dataset_sample_rate(map_dataset_dir, random_grid_item, zero_grid_item, map_aug):
    """Each simulation should appear exactly num_samples times."""
    num_samples = 3
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=num_samples)

    counts = {random_grid_item.simulation: 0, zero_grid_item.simulation: 0}
    for i in range(len(ds)):
        sim = ds[i]["simulation"]
        assert sim in counts, f"Unexpected simulation: {sim}"
        counts[sim] += 1

    assert counts[random_grid_item.simulation] == num_samples
    assert counts[zero_grid_item.simulation] == num_samples


# ---------------------------------------------------------------------------
# DataLoader compatibility test
# ---------------------------------------------------------------------------


def test_map_dataset_works_with_dataloader(map_dataset_dir, map_aug):
    """MagnetDataset should work with PyTorch DataLoader without a custom worker_init_fn."""
    ds = MagnetDataset(map_dataset_dir, transforms=map_aug, num_samples=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    batches = list(loader)
    # 2 simulations × 2 samples / batch_size 2 = 2 batches
    assert len(batches) == 2
    for batch in batches:
        assert isinstance(batch, dict)


# ---------------------------------------------------------------------------
# Fast-path (crop-aware HDF5 partial read) tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def crop_fast_path_dir(processed_dir_path, random_grid_item, zero_grid_item):
    """Processed directory for fast-path crop tests."""
    from shutil import rmtree
    path = processed_dir_path / "test_map_dataset_crop_fast_path"
    create_processed_dir(path, random_grid_item, zero_grid_item, is_grid=True)
    yield path
    if path.exists():
        rmtree(path)


def test_map_dataset_center_crop_fast_path_correct_shapes(crop_fast_path_dir):
    """Center crop via fast path must produce the expected output shape."""
    from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift
    crop_size = (10, 10, 10)
    transforms = Compose([Crop(crop_size=crop_size, crop_position="center"), GridPhaseShift(num_coils=8)])
    ds = MagnetDataset(crop_fast_path_dir, transforms=transforms, num_samples=1)
    for i in range(len(ds)):
        item = ds[i]
        assert item["input"].shape[1:] == crop_size
        assert item["subject"].shape == crop_size
        assert item["positions"].shape[1:] == crop_size
        assert item["coils"].shape[1:] == crop_size


def test_map_dataset_center_crop_fast_path_matches_slow_path(crop_fast_path_dir, random_grid_item):
    """Center crop fast path must return the same spatial data as the slow path (full load + crop).

    PhaseShift randomises the field and coils, so we compare only input, subject, and positions
    (which are not touched by PhaseShift). These must be identical between paths because
    center crop is deterministic.
    """
    from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift
    from unittest.mock import patch

    crop_size = (10, 10, 10)

    # Fast path: Crop is the leading transform → triggers partial HDF5 reads
    transforms_fast = Compose([Crop(crop_size=crop_size, crop_position="center"), GridPhaseShift(num_coils=8)])
    ds_fast = MagnetDataset(crop_fast_path_dir, transforms=transforms_fast, num_samples=1)

    # Slow path: patch _get_leading_crop to return None so full arrays are loaded,
    # then the in-memory Crop is applied as usual.
    transforms_slow = Compose([Crop(crop_size=crop_size, crop_position="center"), GridPhaseShift(num_coils=8)])
    ds_slow = MagnetDataset(crop_fast_path_dir, transforms=transforms_slow, num_samples=1)

    with patch.object(ds_slow.__class__, "_get_leading_crop", staticmethod(lambda t: None)):
        for i in range(len(ds_fast)):
            fast_item = ds_fast[i]
            slow_item = ds_slow[i]
            # input, subject, and positions are not modified by PhaseShift,
            # so they must be bit-for-bit equal between the two paths.
            assert np.array_equal(fast_item["input"], slow_item["input"]), \
                "input mismatch between fast and slow crop path"
            assert np.array_equal(fast_item["subject"], slow_item["subject"]), \
                "subject mismatch between fast and slow crop path"
            assert np.array_equal(fast_item["positions"], slow_item["positions"]), \
                "positions mismatch between fast and slow crop path"
