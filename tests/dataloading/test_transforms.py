import pytest

from magnet_pinn.data.transforms import Compose


def test_compose_none_transformations_given():
    with pytest.raises(ValueError):
        transforms = Compose(None)


def test_compose_empty_list():
    with pytest.raises(ValueError):
        transforms = Compose([])


def test_compose_with_none_transformations():
    with pytest.raises(ValueError):
        transforms = Compose([None, None])

    
def test_compose_with_invalid_type_transform():
    with pytest.raises(ValueError):
        transforms = Compose(["transformation"])