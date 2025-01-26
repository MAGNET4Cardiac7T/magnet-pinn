import pytest

import numpy as np
from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.data.transforms import (
    Compose, Crop, GridPhaseShift, PointPhaseShift, PointSampling, 
    PhaseShift, BaseTransform, DefaultTransform
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
    check_items_datatypes(result, random_item)


def test_crop_transform_crop_check_datatypes_random_crop(random_item):
    crop_random = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = crop_random(random_item)
    check_items_datatypes(result, random_item)


def check_items_datatypes(result, random_item):
    assert type(result.simulation) == type(random_item.simulation)
    assert result.input.dtype == random_item.input.dtype
    assert result.field.dtype == random_item.field.dtype
    assert result.subject.dtype == random_item.subject.dtype
    assert type(result.positions) == type(random_item.positions)
    assert result.phase.dtype == random_item.phase.dtype
    assert result.mask.dtype == random_item.mask.dtype
    assert result.coils.dtype == random_item.coils.dtype
    assert result.dtype == random_item.dtype
    assert type(result.dtype) == type(random_item.dtype)
    assert result.truncation_coefficients.dtype == random_item.truncation_coefficients.dtype


def test_crop_transform_valid_central_crop_position_shape(zero_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(zero_item)
    check_cropped_shapes(result)


def test_crop_transform_valid_random_crop_position_shape(zero_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = augment(zero_item)
    check_cropped_shapes(result)


def check_cropped_shapes(result):
    assert result.input.shape == (3, 10, 10, 10)
    assert result.field.shape == (2, 2, 3, 10, 10, 10, 8)
    assert result.subject.shape == (10, 10, 10)
    assert len(result.positions) == 0
    assert result.phase.shape == (8,)
    assert result.mask.shape == (8,)
    assert result.coils.shape == (10, 10, 10, 8)
    assert result.truncation_coefficients.shape == (3,)


def test_crop_transform_crop_size_matches_original_central_crop_position(zero_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="center")
    result = crop(zero_item)
    check_items_shapes_suppsed_to_be_equal(result, zero_item)


def test_crop_transform_crop_size_matches_original_random_crop_position(zero_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="random")
    result = crop(zero_item)
    check_items_shapes_suppsed_to_be_equal(result, zero_item)


def check_items_shapes_suppsed_to_be_equal(result, input_item):
    assert result.input.shape == input_item.input.shape
    assert result.field.shape == input_item.field.shape
    assert result.subject.shape == input_item.subject.shape
    assert len(result.positions) == len(input_item.positions)
    assert result.phase.shape == input_item.phase.shape
    assert result.mask.shape == input_item.mask.shape
    assert result.coils.shape == input_item.coils.shape
    assert result.truncation_coefficients.shape == input_item.truncation_coefficients.shape


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
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(21, 10, 10), crop_position="center")(zero_item)
        _ = Crop(crop_size=(10, 21, 10), crop_position="center")(zero_item)
        _ = Crop(crop_size=(10, 10, 21), crop_position="center")(zero_item)


def test_crop_transform_crop_size_axis_bigger_than_original_random_crop_position(zero_item):
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(21, 10, 10), crop_position="random")(zero_item)
        _ = Crop(crop_size=(10, 21, 10), crop_position="random")(zero_item)
        _ = Crop(crop_size=(10, 10, 21), crop_position="random")(zero_item)


def test_crop_transform_valid_central_crop_position_check_values(random_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(random_item)

    check_elements_not_changed_by_crop(result, random_item)
    assert np.equal(result.input, random_item.input[:, 5:15, 5:15, 5:15]).all()
    assert np.equal(result.field, random_item.field[:, :, :, 5:15, 5:15, 5:15, :]).all()
    assert np.equal(result.subject, random_item.subject[5:15, 5:15, 5:15]).all()
    assert np.equal(result.coils, random_item.coils[5:15, 5:15, 5:15, :]).all()


def test_crop_transform_valid_random_crop_position(zero_item):
    """
    As a test array we take zeros array, so the cropped array would be also zeros array
    """
    crop = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = crop(zero_item)

    check_elements_not_changed_by_crop(result, zero_item)
    assert np.equal(result.field, zero_item.field[:, :, :, 0:10, 0:10, 0:10, :]).all()
    assert np.equal(result.input, zero_item.input[:, 0:10, 0:10, 0:10]).all()
    assert np.equal(result.subject, zero_item.subject[0:10, 0:10, 0:10]).all()
    assert np.equal(result.coils, zero_item.coils[0:10, 0:10, 0:10, :]).all()


def check_elements_not_changed_by_crop(result, input_item):
    assert result.simulation == input_item.simulation
    assert result.positions == input_item.positions
    assert np.equal(result.phase, input_item.phase).all()
    assert np.equal(result.mask, input_item.mask).all()
    assert result.dtype == input_item.dtype
    assert np.equal(result.truncation_coefficients, input_item.truncation_coefficients).all()


def test_crop_transform_actions_not_inplace(zero_item):
    crop = Crop(crop_size=(10, 10, 10))
    result = crop(zero_item)

    assert result is not zero_item


def test_default_transform_invalid_dataitem():
    with pytest.raises(ValueError):
        _ = DefaultTransform()(None)


def test_default_transform_actions_not_inplace(zero_item):
    trans = DefaultTransform()
    result = trans(zero_item)

    assert result is not zero_item


def test_default_transform_check_datatypes(zero_item):
    result = DefaultTransform()(zero_item)
    check_items_datatypes(result, zero_item)


def test_default_transform_check_shapes(zero_item):
    result = DefaultTransform()(zero_item)

    assert result.input.shape == zero_item.input.shape
    assert result.subject.shape == zero_item.subject.shape
    assert len(result.positions) == len(zero_item.positions)
    assert result.phase.shape == zero_item.phase.shape
    assert result.mask.shape == zero_item.mask.shape

    assert result.field.shape == zero_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(zero_item.coils.shape[:-1]))


def test_default_transform_check_values(random_item):
    result = DefaultTransform()(random_item)

    assert result.simulation == random_item.simulation
    assert np.equal(result.input, random_item.input).all()
    assert np.equal(result.subject, random_item.subject).all()
    assert result.positions == random_item.positions
    assert result.dtype == random_item.dtype
    assert np.equal(result.truncation_coefficients, random_item.truncation_coefficients).all()

    assert np.equal(result.field, np.sum(random_item.field, axis=-1)).all()

    coils_num = random_item.coils.shape[-1]
    assert np.equal(result.phase, np.zeros(coils_num, dtype=random_item.phase.dtype)).all()
    assert np.equal(result.mask, np.ones(coils_num, dtype=random_item.mask.dtype)).all()

    expected_coils = np.stack([
        np.sum(random_item.coils, axis=-1),
        np.zeros(random_item.coils.shape[:-1], dtype=random_item.coils.dtype)
    ], axis=0)
    assert np.equal(result.coils, expected_coils).all()


def test_phase_shift_transform_check_properties_uniform():
    aug = PhaseShift(num_coils=8, sampling_method="uniform")

    assert aug.num_coils == 8
    assert aug.sampling_method == "uniform"


def test_phase_shift_transform_check_properties_binomial():
    aug = PhaseShift(num_coils=8, sampling_method="binomial")

    assert aug.num_coils == 8
    assert aug.sampling_method == "binomial"


def test_phase_shift_transform_check_properties_invalid_sampling_method():
    with pytest.raises(ValueError):
        _ = PhaseShift(num_coils=8, sampling_method="invalid")


def test_phase_shift_transform_check_invalid_simulation():
    with pytest.raises(ValueError):
        _ = PhaseShift(num_coils=8, sampling_method="uniform")(None)


def test_phase_shift_transform_check_valid_processing_dtypes_uniform(random_item):
    aug = PhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_item)

    check_items_datatypes(result, random_item)


def test_phase_shift_transform_check_valid_processing_dtypes_binomial(random_item):
    """
    Indeed the data item does not have a phase property in the normal case before entering the
    phase shifter, but we have anyway created it in the item fixture for easy check later
    """
    aug = PhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_item)

    check_items_datatypes(result, random_item)


def test_phase_shift_transform_check_valid_processing_shapes_uniform(random_item):
    aug = PhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_item)

    check_constant_shapes_not_changed_by_phase_shift(result, random_item)
    assert result.field.shape == random_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_item.coils.shape[:-1]))


def check_constant_shapes_not_changed_by_phase_shift(result, item): 
    assert len(result.simulation) == len(item.simulation)
    assert result.input.shape == item.input.shape
    assert result.subject.shape == item.subject.shape
    assert len(result.positions) == len(item.positions)
    assert result.phase.shape == item.phase.shape
    assert result.mask.shape == item.mask.shape
    assert len(result.dtype) == len(item.dtype)
    assert result.truncation_coefficients.shape == item.truncation_coefficients.shape


def test_phase_shift_transform_check_valid_processing_shapes_binomial(random_item):
    aug = PhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_item)

    check_constant_shapes_not_changed_by_phase_shift(result, random_item)
    assert result.field.shape == random_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_item.coils.shape[:-1]))


def test_phase_shift_transform_check_values_uniform(random_item):
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_item)

    check_constant_values_not_changed_by_phase_shift(result, random_item)
    assert not np.equal(result.phase, random_item.phase).all()
    assert not np.equal(result.mask, random_item.mask).all()


def check_constant_values_not_changed_by_phase_shift(result, item):
    assert result.simulation == item.simulation
    assert np.equal(result.input, item.input).all()
    assert np.equal(result.subject, item.subject).all()
    assert result.positions == item.positions
    assert result.dtype == item.dtype
    assert np.equal(result.truncation_coefficients, item.truncation_coefficients).all()


def test_phase_shift_transform_check_values_binomial(random_item):
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_item)

    check_constant_values_not_changed_by_phase_shift(result, random_item)
    assert not np.equal(result.phase, random_item.phase).all()
    assert not np.equal(result.mask, random_item.mask).all()
