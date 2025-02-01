from collections.abc import Iterable

from magnet_pinn.data._base import MagnetBaseIterator
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.point import MagnetPointIterator


def test_base_iterator_is_iterator():
    assert issubclass(MagnetBaseIterator, Iterable)


def test_grid_dataset_is_iterator():
    assert issubclass(MagnetGridIterator, MagnetBaseIterator)


def test_point_dataset_is_iterator():
    assert issubclass(MagnetPointIterator, MagnetBaseIterator)