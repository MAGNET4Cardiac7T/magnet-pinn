import pytest

from magnet_pinn.data.transforms import Compose
import numpy as np
from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift, PointPhaseShift, PointSampling, PhaseShift
from magnet_pinn.data.dataitem import DataItem


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


def test_crop():
    data = DataItem(
        input=np.random.rand(1, 10, 10, 10),
        subject=np.random.rand(10, 10, 10),
        simulation=np.random.rand(10, 10, 10),
        field=np.random.rand(2, 2, 10, 10, 10),
        phase=np.random.rand(10, 10, 10),
        mask=np.random.rand(10, 10, 10),
        coils=np.random.rand(10, 10, 10),
        dtype='float32',
        truncation_coefficients=np.random.rand(10),
    )
    crop = Crop(crop_size=(5, 5, 5))
    cropped_data = crop(data)
    assert cropped_data.input.shape == (1, 5, 5, 5)
    assert cropped_data.subject.shape == (5, 5, 5)
    assert cropped_data.field.shape == (2, 2, 5, 5, 5)


def test_grid_phase_shift():
    data = DataItem(
        input=np.random.rand(1, 10, 10, 10),
        subject=np.random.rand(10, 10, 10),
        simulation=np.random.rand(10, 10, 10),
        field=np.random.rand(2, 2, 10, 10, 10),
        phase=np.random.rand(10, 10, 10),
        mask=np.random.rand(10, 10, 10),
        coils=np.random.rand(10, 10, 10),
        dtype='float32',
        truncation_coefficients=np.random.rand(10),
    )
    phase_shift = GridPhaseShift(num_coils=10)
    shifted_data = phase_shift(data)
    assert shifted_data.field.shape == data.field.shape
    assert shifted_data.coils.shape == data.coils.shape


def test_point_phase_shift():
    data = DataItem(
        input=np.random.rand(1, 10, 10, 10),
        subject=np.random.rand(10, 10, 10),
        simulation=np.random.rand(10, 10, 10),
        field=np.random.rand(2, 2, 10, 10, 10),
        phase=np.random.rand(10, 10, 10),
        mask=np.random.rand(10, 10, 10),
        coils=np.random.rand(10, 10, 10),
        dtype='float32',
        truncation_coefficients=np.random.rand(10),
    )
    phase_shift = PointPhaseShift(num_coils=10)
    shifted_data = phase_shift(data)
    assert shifted_data.field.shape == data.field.shape
    assert shifted_data.coils.shape == data.coils.shape


def test_point_sampling():
    data = DataItem(
        input=np.random.rand(100, 10),
        subject=np.random.rand(100, 10),
        simulation=np.random.rand(100, 10),
        field=np.random.rand(2, 2, 100),
        phase=np.random.rand(100, 10),
        mask=np.random.rand(100, 10),
        coils=np.random.rand(100, 10),
        dtype='float32',
        truncation_coefficients=np.random.rand(10),
        positions=np.random.rand(100, 3)
    )
    point_sampling = PointSampling(points_sampled=0.5)
    sampled_data = point_sampling(data)
    assert sampled_data.input.shape[0] == 50
    assert sampled_data.subject.shape[0] == 50
    assert sampled_data.field.shape[2] == 50
    assert sampled_data.positions.shape[0] == 50


def test_phase_shift():
    data = DataItem(
        input=np.random.rand(1, 10, 10, 10),
        subject=np.random.rand(10, 10, 10),
        simulation=np.random.rand(10, 10, 10),
        field=np.random.rand(2, 2, 10, 10, 10),
        phase=np.random.rand(10, 10, 10),
        mask=np.random.rand(10, 10, 10),
        coils=np.random.rand(10, 10, 10),
        dtype='float32',
        truncation_coefficients=np.random.rand(10),
    )
    phase_shift = PhaseShift(num_coils=10)
    shifted_data = phase_shift(data)
    assert shifted_data.field.shape == data.field.shape
    assert shifted_data.coils.shape == data.coils.shape
