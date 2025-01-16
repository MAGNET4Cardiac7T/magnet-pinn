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


def test_compose_running_order():
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

    item = DataItem(
        simulation="",
        input=np.array([]),
        subject=np.array([]),
        field=np.array([])
    )

    result_item = aug(item)

    assert result_item.simulation == "123"


def test_crop_transform_crop_size_is_none():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=None)


def test_crop_transform_crop_size_with_invalid_dimension():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(100, 100))

    with pytest.raises(ValueError):
        _ = Crop(crop_size=(100, 100, 100, 100))


def test_crop_transform_crop_size_is_invalid_type():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, 0, 3.5))

    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, "value", 0))


def test_crop_transformt_crop_position_invalid_type():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, 1, 1), crop_position="value")


def test_crop_transform_test_result_size():
    data = DataItem(
        simulation="",
        field=np.zeros((2, 2, 3, 20, 20, 20, 8)),
        input=np.zeros((3, 20, 20, 20)),
        subject=np.zeros((20, 20, 20)),
        coils=np.zeros((20, 20, 20, 8))
    )

    augment = Crop(crop_size=(10, 10, 10))

    result = augment(data)

    assert result.field.shape == (2, 2, 3, 10, 10, 10, 8)
    assert result.input.shape == (3, 10, 10, 10)
    assert result.subject.shape == (10, 10, 10)
    assert result.coils.shape == (10, 10, 10, 8)


def test_crop_transform_compare_starting_positions():
    data = DataItem(
        simulation="",
        field=np.random.rand(2, 2, 3, 20, 20, 20, 8),
        input=np.random.rand(3, 20, 20, 20),
        subject=np.random.rand(20, 20, 20),
        coils=np.random.rand(20, 20, 20, 8)
    )

    central_crop = Crop(crop_size=(10, 10, 10), crop_position="center")
    random_crop = Crop(crop_size=(10, 10, 10), crop_position="random")

    central_cropped = central_crop(data)
    randomly_cropped = random_crop(data)

    assert not np.equal(central_cropped.field, randomly_cropped.field).all()
    assert not np.equal(central_cropped.input, randomly_cropped.input).all()
    assert not np.equal(central_cropped.subject, randomly_cropped.subject).all()
    assert not np.equal(central_cropped.coils, randomly_cropped.coils).all()
