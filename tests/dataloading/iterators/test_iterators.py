from collections.abc import Iterable
from shutil import rmtree

import pytest

from magnet_pinn.data._base import MagnetBaseIterator
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.point import MagnetPointIterator
from magnet_pinn.data.transforms import DefaultTransform
from magnet_pinn.preprocessing.preprocessing import (
    PROCESSED_ANTENNA_DIR_PATH, TARGET_FILE_NAME
)


def test_base_iterator_is_iterator():
    assert issubclass(MagnetBaseIterator, Iterable)


def test_grid_dataset_is_iterator():
    assert issubclass(MagnetGridIterator, MagnetBaseIterator)


def test_point_dataset_is_iterator():
    assert issubclass(MagnetPointIterator, MagnetBaseIterator)


def test_grid_iterator_not_existing_coils_dir(grid_prodcessed_dir_short_term):
    antenna_dir_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    rmtree(antenna_dir_path)

    aug = DefaultTransform()
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_not_existing_coils_file(grid_prodcessed_dir_short_term):
    antenna_file_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    antenna_file_path.unlink()

    aug = DefaultTransform()
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_invalid_antenna_dir(grid_prodcessed_dir_short_term):
    antenna_dir_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH
    dir_changed_path = antenna_dir_path.with_name("changed")
    antenna_dir_path.rename(dir_changed_path)

    aug = DefaultTransform()
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)


def test_grid_iterator_invalid_antenna_file_name(grid_prodcessed_dir_short_term):
    antenna_file_path = grid_prodcessed_dir_short_term / PROCESSED_ANTENNA_DIR_PATH / TARGET_FILE_NAME.format(name="antenna")
    file_changed_path = antenna_file_path.with_name("changed")
    antenna_file_path.rename(file_changed_path)

    aug = DefaultTransform()
    with pytest.raises(FileNotFoundError):
        _ = MagnetGridIterator(grid_prodcessed_dir_short_term, transforms=aug, num_samples=1)
