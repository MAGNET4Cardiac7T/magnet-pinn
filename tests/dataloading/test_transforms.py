import pytest

import numpy as np
from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.data.transforms import (
    Compose, Crop, GridPhaseShift, PointPhaseShift, PointSampling, PhaseShift, BaseTransform
)


def test_compose_none_transformations_given():
    with pytest.raises(ValueError):
        _ = Compose(None)


def test_compose_empty_list():
    with pytest.raises(ValueError):
        _ = Compose([])


def test_compose_with_none_transformations():
    with pytest.raises(ValueError):
        transforms = Compose([None, None])

    
def test_compose_with_invalid_type_transform():
    with pytest.raises(ValueError):
        transforms = Compose(["transformation"])


def test_compose_running_order(zero_item):
    class FirstAugmentation(BaseTransform):
        def __call__(self, simulation: DataItem) -> DataItem:
            simulation.simulation += "1"
            return simulation


    class SecondAugmentation(BaseTransform):
        def __call__(self, simulation: DataItem) -> DataItem:
            simulation.simulation += "2"
            return simulation


    class ThirdAugmentation(BaseTransform):
        def __call__(self, simulation: DataItem) -> DataItem:
            simulation.simulation += "3"
            return simulation

    
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])

    result_item = aug(zero_item)

    assert result_item.simulation == "123"


def test_crop_transform_crop_size_none():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=None)


def test_crop_transform_crop_size_invalid_dimensions_number():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(100, 100))

    with pytest.raises(ValueError):
        _ = Crop(crop_size=(100, 100, 100, 100))


def test_crop_transform_crop_size_invalid_type():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, 0, 3.5))

    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, "value", 0))


def test_crop_transform_crop_position_invalid_type():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, 1, 1), crop_position="value")


def test_crop_transform_crop_check_datatypes_central_crop(random_item):
    crop_central = Crop(crop_size=(10, 10, 10), crop_position="center")

    result = crop_central(random_item)

    assert result.field.dtype == random_item.field.dtype
    assert result.input.dtype == random_item.input.dtype
    assert result.subject.dtype == random_item.subject.dtype
    assert result.coils.dtype == random_item.coils.dtype


def test_crop_transform_crop_check_datatypes_random_crop(random_item):
    crop_random = Crop(crop_size=(10, 10, 10), crop_position="random")

    result = crop_random(random_item)

    assert result.field.dtype == random_item.field.dtype
    assert result.input.dtype == random_item.input.dtype
    assert result.subject.dtype == random_item.subject.dtype
    assert result.coils.dtype == random_item.coils.dtype


def test_crop_transform_valid_central_crop_position_shape(zero_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(zero_item)

    assert result.field.shape == (2, 2, 3, 10, 10, 10, 8)
    assert result.input.shape == (3, 10, 10, 10)
    assert result.subject.shape == (10, 10, 10)
    assert result.coils.shape == (10, 10, 10, 8)


def test_crop_transform_valid_random_crop_position_shape(zero_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = augment(zero_item)

    assert result.field.shape == (2, 2, 3, 10, 10, 10, 8)
    assert result.input.shape == (3, 10, 10, 10)
    assert result.subject.shape == (10, 10, 10)
    assert result.coils.shape == (10, 10, 10, 8)


def test_crop_transform_crop_size_matches_original_central_crop_position(zero_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="center")

    result = crop(zero_item)

    assert result.field.shape == zero_item.field.shape
    assert result.input.shape == zero_item.input.shape
    assert result.subject.shape == zero_item.subject.shape
    assert result.coils.shape == zero_item.coils.shape


def test_crop_transform_crop_size_matches_original_random_crop_position(zero_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="random")

    result = crop(zero_item)

    assert result.field.shape == zero_item.field.shape
    assert result.input.shape == zero_item.input.shape
    assert result.subject.shape == zero_item.subject.shape
    assert result.coils.shape == zero_item.coils.shape


def test_crop_transform_crop_size_axis_less_equal_zero():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(0, 10, 10), crop_position="center")
        _ = Crop(crop_size=(-1, 10, 10), crop_position="center")
        _ = Crop(crop_size=(0, 10, 10), crop_position="random")
        _ = Crop(crop_size=(-1, 10, 10), crop_position="random")
        _ = Crop(crop_size=(10, 0, 10), crop_position="center")
        _ = Crop(crop_size=(10, -1, 10), crop_position="center")
        _ = Crop(crop_size=(10, 0, 10), crop_position="random")
        _ = Crop(crop_size=(10, -1, 10), crop_position="random")
        _ = Crop(crop_size=(10, 10, 0), crop_position="center")
        _ = Crop(crop_size=(10, 10, -1), crop_position="center")
        _ = Crop(crop_size=(10, 10, 0), crop_position="random")
        _ = Crop(crop_size=(10, 10, -1), crop_position="random")


def test_crop_transform_crop_size_axis_bigger_than_original_central_crop_position(zero_item):
    crop_x = Crop(crop_size=(21, 10, 10), crop_position="center")
    crop_y = Crop(crop_size=(10, 21, 10), crop_position="center")
    crop_z = Crop(crop_size=(10, 10, 21), crop_position="center")
    with pytest.raises(ValueError):
        _ = crop_x(zero_item)
        _ = crop_y(zero_item)
        _ = crop_z(zero_item)


def test_crop_transform_crop_size_axis_bigger_than_original_random_crop_position(zero_item):
    crop_x = Crop(crop_size=(21, 10, 10), crop_position="random")
    crop_y = Crop(crop_size=(10, 21, 10), crop_position="random")
    crop_z = Crop(crop_size=(10, 10, 21), crop_position="random")
    with pytest.raises(ValueError):
        _ = crop_x(zero_item)
        _ = crop_y(zero_item)
        _ = crop_z(zero_item)


def test_crop_transform_valid_central_crop_position_check_values(random_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(random_item)

    assert np.equal(result.field, random_item.field[:, :, :, 5:15, 5:15, 5:15, :]).all()
    assert np.equal(result.input, random_item.input[:, 5:15, 5:15, 5:15]).all()
    assert np.equal(result.subject, random_item.subject[5:15, 5:15, 5:15]).all()
    assert np.equal(result.coils, random_item.coils[5:15, 5:15, 5:15, :]).all()


def test_crop_transform_valid_random_crop_position(zero_item):
    """
    As a test array we take zeros array, so the cropped array would be also zeros array
    """
    crop = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = crop(zero_item)

    assert np.equal(result.field, zero_item.field[:, :, :, 0:10, 0:10, 0:10, :]).all()
    assert np.equal(result.input, zero_item.input[:, 0:10, 0:10, 0:10]).all()
    assert np.equal(result.subject, zero_item.subject[0:10, 0:10, 0:10]).all()
    assert np.equal(result.coils, zero_item.coils[0:10, 0:10, 0:10, :]).all()
