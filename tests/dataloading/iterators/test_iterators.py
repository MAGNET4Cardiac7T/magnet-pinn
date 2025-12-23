from shutil import rmtree
from collections.abc import Iterable

import pytest
import numpy as np

from magnet_pinn.data._base import MagnetBaseIterator
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.preprocessing.preprocessing import (
    PROCESSED_ANTENNA_DIR_PATH, TARGET_FILE_NAME, PROCESSED_SIMULATIONS_DIR_PATH
)
from tests.dataloading.iterators.helpers import (
    RANDOM_SIM_FILE_NAME, ZERO_SIM_FILE_NAME
)
from tests.dataloading.iterators.helpers import (
    check_dtypes_between_iter_result_and_supposed_simulation,
    check_shapes_between_item_result_and_supposed_simulation,
    check_values_between_item_result_and_supposed_simulation,
    check_shapes_between_item_result_and_supposed_simulation_for_pointclous
)


def test_base_iterator_is_iterator():
    assert issubclass(MagnetBaseIterator, Iterable)


def test_base_iterator_can_be_instantiated(grid_processed_dir, grid_aug):
    """Test that MagnetBaseIterator can be directly instantiated since it has no abstract methods"""
    iterator = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    assert isinstance(iterator, MagnetBaseIterator)


def test_base_iterator_check_coils_properties(grid_processed_dir, random_grid_item, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)

    expected_coils_path = grid_processed_dir / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    assert iter.coils_path == expected_coils_path
    assert iter.num_coils == random_grid_item.coils.shape[0]

    assert iter.coils.shape == random_grid_item.coils.shape
    assert iter.coils.dtype == np.bool_
    assert np.equal(iter.coils, random_grid_item.coils.astype(np.bool_)).all()


def test_base_iterator_check_simulations_properties(grid_processed_dir, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)

    expected_sim_dir = grid_processed_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert iter.simulation_dir == expected_sim_dir

    expected_sim_list = [
        expected_sim_dir / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME),
        expected_sim_dir / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME)
    ]
    assert iter.simulation_list == expected_sim_list


def test_base_iterator_check_other_properties(grid_processed_dir, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=2)

    assert iter.num_samples == 2
    assert iter.transforms == grid_aug


def test_base_iterator_check_num_samples_eq_to_zero(grid_processed_dir, grid_aug):
    with pytest.raises(ValueError):
        _ = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=0)


def test_base_iterator_check_num_samples_less_than_0(grid_processed_dir, grid_aug):
    with pytest.raises(ValueError):
        _ = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=-1)


def test_base_iterator_check_invalid_transforms(grid_processed_dir):
    with pytest.raises(ValueError):
        _ = MagnetBaseIterator(grid_processed_dir, transforms=None, num_samples=1)


def test_base_iterator_check_overall_samples_numbers_for_unit_num_samples(grid_processed_dir, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    sampled = list(iter)

    assert len(sampled) == 2


def test_base_iterator_check_overall_samples_numbers_for_multiple_num_samples(grid_processed_dir, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=100)
    sampled = list(iter)

    assert len(sampled) == 200


def test_base_iterator_check_sampled_data_items_datatypes(grid_processed_dir, random_grid_item, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    for item in iter:
        check_dtypes_between_iter_result_and_supposed_simulation(item, random_grid_item)


def test_base_iterator_check_sampled_data_items_shapes(grid_processed_dir, random_grid_item, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    for item in iter:
        check_shapes_between_item_result_and_supposed_simulation(item, random_grid_item)


def test_base_iterator_check_sampled_data_items_values(grid_processed_dir, random_grid_item, zero_grid_item, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    for result in iter:
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            assert False, f"Unexpected simulation: {result['simulation']}"


def test_base_iterator_check_sampled_data_rate(grid_processed_dir, random_grid_item, zero_grid_item, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=3)

    random_samples_count = 0
    zero_samples_count = 0
    for result in iter:
        if result["simulation"] == random_grid_item.simulation:
            random_samples_count += 1
        elif result["simulation"] == zero_grid_item.simulation:
            zero_samples_count += 1
        else:
            assert False, f"Unexpected simulation: {result['simulation']}"

    assert random_samples_count == zero_samples_count == 3


def test_base_iterator_check_multiple_samples(grid_processed_dir, random_grid_item, zero_grid_item, grid_aug):
    iter = MagnetBaseIterator(grid_processed_dir, transforms=grid_aug, num_samples=3)

    sampled = list(iter)
    for result in sampled:
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            assert False, f"Unexpected simulation: {result['simulation']}"


def test_grid_dataset_is_iterator():
    assert issubclass(MagnetGridIterator, MagnetBaseIterator)


def test_point_dataset_is_iterator():
    assert issubclass(MagnetPointIterator, MagnetBaseIterator)


def test_grid_iterator_check_not_existing_coils_dir(grid_prodcessed_dir_short_term, grid_aug):
    antenna_dir_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    rmtree(antenna_dir_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_not_existing_coils_file(grid_prodcessed_dir_short_term, grid_aug):
    antenna_file_path = (
        grid_prodcessed_dir_short_term
        / PROCESSED_ANTENNA_DIR_PATH
        / TARGET_FILE_NAME.format(name="antenna")
    )
    antenna_file_path.unlink()

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_invalid_antenna_dir(grid_prodcessed_dir_short_term, grid_aug):
    antenna_dir_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    dir_changed_path = antenna_dir_path.with_name("changed")
    antenna_dir_path.rename(dir_changed_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_invalid_antenna_file_name(grid_prodcessed_dir_short_term, grid_aug):
    antenna_file_path = (
        grid_prodcessed_dir_short_term
        / PROCESSED_ANTENNA_DIR_PATH
        / TARGET_FILE_NAME.format(name="antenna")
    )
    file_changed_path = antenna_file_path.with_name("changed")
    antenna_file_path.rename(file_changed_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_not_existing_simulations_dir(grid_prodcessed_dir_short_term, grid_aug):
    simulations_dir_path = grid_prodcessed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_invalid_simulations_dir(grid_prodcessed_dir_short_term, grid_aug):
    simulations_dir_path = grid_prodcessed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    dir_changed_path = simulations_dir_path.with_name("changed")
    simulations_dir_path.rename(dir_changed_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_empty_simulations_dir(grid_prodcessed_dir_short_term, grid_aug):
    """
    Instead of just deliting all simulations inside we recreate a directory.
    """
    simulations_dir_path = grid_prodcessed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir_path)
    simulations_dir_path.mkdir()

    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=grid_aug, num_samples=1)


def test_grid_iterator_check_coils_properties(grid_processed_dir, random_grid_item, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)

    expected_coils_path = grid_processed_dir / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    assert iter.coils_path == expected_coils_path
    assert iter.num_coils == random_grid_item.coils.shape[0]

    assert iter.coils.shape == random_grid_item.coils.shape
    assert iter.coils.dtype == np.bool_
    assert np.equal(iter.coils, random_grid_item.coils.astype(np.bool_)).all()


def test_grid_iterator_check_simulations_properties(grid_processed_dir, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)

    expected_sim_dir = grid_processed_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert iter.simulation_dir == expected_sim_dir

    expected_sim_list = [
        expected_sim_dir / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME),
        expected_sim_dir / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME)
    ]
    assert iter.simulation_list == expected_sim_list


def test_grid_iterator_check_other_properties(grid_processed_dir, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=2)

    assert iter.num_samples == 2
    assert iter.transforms == grid_aug


def test_grid_iterator_check_num_samples_eq_to_zero(grid_processed_dir, grid_aug):
    with pytest.raises(ValueError):
        _ = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=0)


def test_grid_iterator_check_num_samples_less_than_0(grid_processed_dir, grid_aug):
    with pytest.raises(ValueError):
        _ = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=-1)


def test_grid_iterator_check_invalid_transforms(grid_processed_dir):
    with pytest.raises(ValueError):
        _ = MagnetGridIterator(grid_processed_dir, transforms=None, num_samples=1)


def test_grid_iterator_check_overall_samples_numbers_for_unit_num_samples(grid_processed_dir, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    sampled = list(iter)

    assert len(sampled) == 2


def test_grid_iterator_check_overall_samples_numbers_for_multiple_num_samples(grid_processed_dir, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=100)
    sampled = list(iter)

    assert len(sampled) == 200


def test_grid_iterator_check_sampled_data_items_datatypes(grid_processed_dir, random_grid_item, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    for item in iter:
        check_dtypes_between_iter_result_and_supposed_simulation(item, random_grid_item)


def test_grid_iterator_check_sampled_data_items_shapes(grid_processed_dir, random_grid_item, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    for item in iter:
        check_shapes_between_item_result_and_supposed_simulation(item, random_grid_item)


def test_grid_iterator_check_sampled_data_items_values(grid_processed_dir, random_grid_item, zero_grid_item, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=1)
    for result in iter:
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            raise ValueError("Unexpected simulation name.")


def test_grid_iterator_check_sampled_data_rate(grid_processed_dir, random_grid_item, zero_grid_item, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=3)

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


def test_grid_iteartor_check_multiple_samples(grid_processed_dir, random_grid_item, zero_grid_item, grid_aug):
    iter = MagnetGridIterator(grid_processed_dir, transforms=grid_aug, num_samples=3)

    sampled = list(iter)
    for result in sampled:
        if result["simulation"] == random_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_grid_item)
        elif result["simulation"] == zero_grid_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_grid_item)
        else:
            raise ValueError("Unexpected simulation name.")


def test_point_iterator_check_not_existing_coils_dir(pointcloud_processed_dir_short_term, pointcloud_aug):
    antenna_dir_path = pointcloud_processed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    rmtree(antenna_dir_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_not_existing_coils_file(pointcloud_processed_dir_short_term, pointcloud_aug):
    antenna_file_path = (
        pointcloud_processed_dir_short_term
        / PROCESSED_ANTENNA_DIR_PATH
        / TARGET_FILE_NAME.format(name="antenna")
    )
    antenna_file_path.unlink()

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_invalid_antenna_dir(pointcloud_processed_dir_short_term, pointcloud_aug):
    antenna_dir_path = pointcloud_processed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    dir_changed_path = antenna_dir_path.with_name("changed")
    antenna_dir_path.rename(dir_changed_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_invalid_antenna_file_name(pointcloud_processed_dir_short_term, pointcloud_aug):
    antenna_file_path = (
        pointcloud_processed_dir_short_term
        / PROCESSED_ANTENNA_DIR_PATH
        / TARGET_FILE_NAME.format(name="antenna")
    )
    file_changed_path = antenna_file_path.with_name("changed")
    antenna_file_path.rename(file_changed_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_not_existing_simulations_dir(pointcloud_processed_dir_short_term, pointcloud_aug):
    simulations_dir_path = pointcloud_processed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_invalid_simulations_dir(pointcloud_processed_dir_short_term, pointcloud_aug):
    simulations_dir_path = pointcloud_processed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    dir_changed_path = simulations_dir_path.with_name("changed")
    simulations_dir_path.rename(dir_changed_path)

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_empty_simulations_dir(pointcloud_processed_dir_short_term, pointcloud_aug):
    simulations_dir_path = pointcloud_processed_dir_short_term / PROCESSED_SIMULATIONS_DIR_PATH
    rmtree(simulations_dir_path)
    simulations_dir_path.mkdir()

    with pytest.raises(FileNotFoundError):
        _ = MagnetPointIterator(pointcloud_processed_dir_short_term, transforms=pointcloud_aug, num_samples=1)


def test_point_iterator_check_coils_properties(pointcloud_processed_dir, random_pointcloud_item, pointcloud_aug):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=1)

    expected_coils_path = (
        pointcloud_processed_dir / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    )
    assert iter.coils_path == expected_coils_path
    assert iter.num_coils == random_pointcloud_item.coils.shape[0]

    assert iter.coils.shape == random_pointcloud_item.coils.shape
    assert iter.coils.dtype == np.bool_
    assert np.equal(iter.coils, random_pointcloud_item.coils.astype(np.bool_)).all()


def test_point_iterator_check_simulations_properties(pointcloud_processed_dir, pointcloud_aug):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=1)

    expected_sim_dir = pointcloud_processed_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert iter.simulation_dir == expected_sim_dir

    expected_sim_list = [
        expected_sim_dir / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME),
        expected_sim_dir / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME)
    ]
    assert iter.simulation_list == expected_sim_list


def test_point_iterator_check_other_properties(pointcloud_processed_dir, pointcloud_aug):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=2)

    assert iter.num_samples == 2
    assert iter.transforms == pointcloud_aug


def test_point_iterator_check_num_samples_eq_to_zero(pointcloud_processed_dir, pointcloud_aug):
    with pytest.raises(ValueError):
        _ = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=0)


def test_point_iterator_check_num_samples_less_than_0(pointcloud_processed_dir, pointcloud_aug):
    with pytest.raises(ValueError):
        _ = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=-1)


def test_point_iterator_check_invalid_transforms(pointcloud_processed_dir):
    with pytest.raises(ValueError):
        _ = MagnetPointIterator(pointcloud_processed_dir, transforms=None, num_samples=1)


def test_point_iterator_check_overall_samples_numbers_for_unit_num_samples(pointcloud_processed_dir, pointcloud_aug):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=1)
    sampled = list(iter)

    assert len(sampled) == 2


def test_point_iterator_check_overall_samples_numbers_for_multiple_num_samples(
    pointcloud_processed_dir, pointcloud_aug
):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=100)
    sampled = list(iter)

    assert len(sampled) == 200


def test_point_iterator_check_sampled_data_items_datatypes(
    pointcloud_processed_dir, random_pointcloud_item, pointcloud_aug
):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=1)
    for item in iter:
        check_dtypes_between_iter_result_and_supposed_simulation(item, random_pointcloud_item)


def test_point_iterator_check_sampled_data_items_shapes(
    pointcloud_processed_dir, random_pointcloud_item, pointcloud_aug
):
    """
    Zero and random data items have same data shapes so it is whatsoever which of them to use for the shape check
    """
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=1)
    for item in iter:
        check_shapes_between_item_result_and_supposed_simulation_for_pointclous(item, random_pointcloud_item)


def test_point_iterator_check_sampled_data_items_values(
    pointcloud_processed_dir, random_pointcloud_item, zero_pointcloud_item, pointcloud_aug
):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=1)
    for result in iter:
        if result["simulation"] == random_pointcloud_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_pointcloud_item)
        elif result["simulation"] == zero_pointcloud_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_pointcloud_item)
        else:
            raise ValueError("Unexpected simulation name.")


def test_point_iterator_check_sampled_data_rate(
    pointcloud_processed_dir, random_pointcloud_item, zero_pointcloud_item, pointcloud_aug
):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=3)

    random_samples_count = 0
    zero_samples_count = 0
    for result in iter:
        if result["simulation"] == random_pointcloud_item.simulation:
            random_samples_count += 1
        elif result["simulation"] == zero_pointcloud_item.simulation:
            zero_samples_count += 1
        else:
            raise ValueError("Unexpected simulation name.")

    assert random_samples_count == zero_samples_count == 3


def test_point_iteartor_check_multiple_samples(
    pointcloud_processed_dir, random_pointcloud_item, zero_pointcloud_item, pointcloud_aug
):
    iter = MagnetPointIterator(pointcloud_processed_dir, transforms=pointcloud_aug, num_samples=3)

    sampled = list(iter)
    for result in sampled:
        if result["simulation"] == random_pointcloud_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, random_pointcloud_item)
        elif result["simulation"] == zero_pointcloud_item.simulation:
            check_values_between_item_result_and_supposed_simulation(result, zero_pointcloud_item)
        else:
            raise ValueError("Unexpected simulation name.")
