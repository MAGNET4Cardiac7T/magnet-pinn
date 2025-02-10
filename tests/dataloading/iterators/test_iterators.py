from shutil import rmtree
from collections.abc import Iterable

import pytest
import numpy as np

from magnet_pinn.data._base import MagnetBaseIterator
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.data.transforms import PhaseShift
from magnet_pinn.preprocessing.preprocessing import (
    PROCESSED_ANTENNA_DIR_PATH, TARGET_FILE_NAME, PROCESSED_SIMULATIONS_DIR_PATH
)
from tests.dataloading.iterators.helpers import (
    RANDOM_SIM_FILE_NAME, ZERO_SIM_FILE_NAME
)
from tests.dataloading.iterators.helpers import (
    check_dtypes_between_iter_result_and_supposed_simulation,
    check_shapes_between_item_result_and_supposed_simulation,
    check_values_between_item_result_and_supposed_simulation
)


def test_base_iterator_is_iterator():
    assert issubclass(MagnetBaseIterator, Iterable)


def test_grid_dataset_is_iterator():
    assert issubclass(MagnetGridIterator, MagnetBaseIterator)


def test_point_dataset_is_iterator():
    assert issubclass(MagnetPointIterator, MagnetBaseIterator)


def test_grid_iterator_check_not_existing_coils_dir(grid_prodcessed_dir_short_term):
    antenna_dir_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    rmtree(antenna_dir_path)

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_not_existing_coils_file(grid_prodcessed_dir_short_term):
    antenna_file_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    antenna_file_path.unlink()

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_invalid_antenna_dir(grid_prodcessed_dir_short_term):
    antenna_dir_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    dir_changed_path = antenna_dir_path.with_name("changed")
    antenna_dir_path.rename(dir_changed_path)

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_invalid_antenna_file_name(grid_prodcessed_dir_short_term):
    antenna_file_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    file_changed_path = antenna_file_path.with_name("changed")
    antenna_file_path.rename(file_changed_path)

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_not_existing_simulations_dir(grid_prodcessed_dir_short_term):
    simulations_dir_path = grid_prodcessed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir_path)

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_invalid_simulations_dir(grid_prodcessed_dir_short_term):
    simulations_dir_path = grid_prodcessed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    dir_changed_path = simulations_dir_path.with_name("changed")
    simulations_dir_path.rename(dir_changed_path)

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_empty_simulations_dir(grid_prodcessed_dir_short_term):
    """
    Instead of just deliting all simulations inside we recreate a directory.
    """
    simulations_dir_path = grid_prodcessed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir_path)
    simulations_dir_path.mkdir()

    aug = PhaseShift(num_coils=8)
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_check_coils_properties(grid_processed_dir, random_grid_item):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)

    expected_coils_path = grid_processed_dir / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    assert iter.coils_path == expected_coils_path
    assert iter.num_coils == random_grid_item.coils.shape[-1]

    assert iter.coils.shape == random_grid_item.coils.shape
    assert iter.coils.dtype == np.bool_
    assert np.equal(iter.coils, random_grid_item.coils.astype(np.bool_)).all()


def test_grid_iterator_check_simulations_properties(grid_processed_dir):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)

    expected_sim_dir = grid_processed_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert iter.simulation_dir == expected_sim_dir

    expected_sim_list = [
        expected_sim_dir / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME),
        expected_sim_dir / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME)
    ]
    assert iter.simulation_list == expected_sim_list


def test_grid_iterator_check_other_properties(grid_processed_dir):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=2)

    assert iter.num_samples == 2
    assert iter.transforms == aug


def test_grid_iterator_check_num_samples_eq_to_zero(grid_processed_dir):
    aug = PhaseShift(num_coils=8)
    with pytest.raises(ValueError):
        _ = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=0)


def test_grid_iterator_check_num_samples_less_than_0(grid_processed_dir):
    aug = PhaseShift(num_coils=8)
    with pytest.raises(ValueError):
        _ = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=-1)


def test_grid_iterator_check_invalid_transforms(grid_processed_dir):
    aug = None
    with pytest.raises(ValueError):
        _ = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)


def test_grid_iterator_check_overall_samples_numbers_for_unit_num_samples(grid_processed_dir):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)
    sampled = list(iter)

    assert len(sampled) == 2


def test_grid_iterator_check_overall_samples_numbers_for_multiple_num_samples(grid_processed_dir):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=100)
    sampled = list(iter)

    assert len(sampled) == 200


def test_grid_iterator_check_sampled_data_items_datatypes(grid_processed_dir, random_grid_item):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)
    sampled = list(iter)
    random_processed_item, zero_processed_iter = sampled
    check_dtypes_between_iter_result_and_supposed_simulation(random_processed_item, random_grid_item)
    check_dtypes_between_iter_result_and_supposed_simulation(zero_processed_iter, random_grid_item)


def test_grid_iterator_check_sampled_data_items_shapes(grid_processed_dir, random_grid_item):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)
    sampled = list(iter)
    random_processed_item, zero_processed_iter = sampled
    check_shapes_between_item_result_and_supposed_simulation(random_processed_item, random_grid_item)
    check_shapes_between_item_result_and_supposed_simulation(zero_processed_iter, random_grid_item)


def test_grid_iterator_check_sampled_data_items_values(grid_processed_dir, random_grid_item, zero_grid_item):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=1)
    for result in iter:
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            raise ValueError("Unexpected simulation name.")


def test_grid_iterator_check_sampled_data_rate(grid_processed_dir, random_grid_item, zero_grid_item):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=3)

    random_samples_count = 0
    zero_samples_count = 0
    for result in iter:
        if result["simulation"] == random_grid_item.simulation:
            random_samples_count += 1
        elif result["simulation"] == zero_grid_item.simulation:
            zero_samples_count += 1
        else:
            raise ValueError("Unexpected simulation name.")

    assert random_samples_count == zero_samples_count == 3


def test_grid_iteartor_check_multiple_samples(grid_processed_dir, random_grid_item, zero_grid_item):
    aug = PhaseShift(num_coils=8)
    iter = MagnetGridIterator(grid_processed_dir, transforms=aug, num_samples=3)

    sampled = list(iter)
    for result in sampled:
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            raise ValueError("Unexpected simulation name.")
