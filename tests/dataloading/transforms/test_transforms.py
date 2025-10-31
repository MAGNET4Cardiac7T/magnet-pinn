import pytest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.data.transforms import (
    Compose, Crop, GridPhaseShift, PointPhaseShift, PointSampling, 
    PhaseShift, BaseTransform, DefaultTransform, Rotate, Mirror
)
from tests.dataloading.transforms.helpers import (
    FirstAugmentation, SecondAugmentation, ThirdAugmentation, check_items_datatypes,
    check_cropped_shapes, check_items_shapes_supposed_to_be_equal, check_elements_not_changed_by_crop,
    check_constant_shapes_not_changed_except_for_field_coils, check_constant_values_not_changed_by_phase_shift,
    check_default_transform_resulting_shapes, check_default_transform_resulting_values,
    check_complex_number_calculations_in_phase_shift, check_complex_number_calculations_in_pointscloud_phase_shift
)


def check_base_transform_callable():
    assert callable(BaseTransform)


def test_compose_none_transformations_given():
    with pytest.raises(ValueError):
        _ = Compose(None)


def test_compose_empty_list():
    with pytest.raises(ValueError):
        _ = Compose([])


def test_compose_with_none_transformations():
    with pytest.raises(ValueError):
        _ = Compose([None, None])

    
def test_compose_with_invalid_type_transform():
    with pytest.raises(ValueError):
        _ = Compose(["transformation"])


def test_compose_running_order_for_grid(zero_grid_item):
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])
    result_item = aug(zero_grid_item)

    assert result_item.simulation == zero_grid_item.simulation + "123"


def test_compose_running_order_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])
    result_item = aug(random_pointcloud_item)

    assert result_item.simulation == random_pointcloud_item.simulation + "123"


def test_compose_transform_not_inplace_processing_for_grid(zero_grid_item):
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])
    result_item = aug(zero_grid_item)

    assert result_item is not zero_grid_item


def test_compose_transform_not_inplace_processing_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])
    result_item = aug(random_pointcloud_item)

    assert result_item is not random_pointcloud_item


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


def test_crop_transform_crop_check_datatypes_central_crop(random_grid_item):
    crop_central = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = crop_central(random_grid_item)
    check_items_datatypes(result, random_grid_item)


def test_crop_transform_crop_check_datatypes_random_crop(random_grid_item):
    crop_random = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = crop_random(random_grid_item)
    check_items_datatypes(result, random_grid_item)


def test_crop_transform_valid_central_crop_position_shape(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(zero_grid_item)
    check_cropped_shapes(result)


def test_crop_transform_valid_random_crop_position_shape(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    augment = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = augment(zero_grid_item)
    check_cropped_shapes(result)


def test_crop_transform_crop_size_matches_original_central_crop_position(zero_grid_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="center")
    result = crop(zero_grid_item)
    check_items_shapes_supposed_to_be_equal(result, zero_grid_item)


def test_crop_transform_crop_size_matches_original_random_crop_position(zero_grid_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="random")
    result = crop(zero_grid_item)
    check_items_shapes_supposed_to_be_equal(result, zero_grid_item)


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


def test_crop_transform_crop_size_axis_bigger_than_original_central_crop_position(zero_grid_item):
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(21, 10, 10), crop_position="center")(zero_grid_item)
        _ = Crop(crop_size=(10, 21, 10), crop_position="center")(zero_grid_item)
        _ = Crop(crop_size=(10, 10, 21), crop_position="center")(zero_grid_item)


def test_crop_transform_crop_size_axis_bigger_than_original_random_crop_position(zero_grid_item):
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(21, 10, 10), crop_position="random")(zero_grid_item)
        _ = Crop(crop_size=(10, 21, 10), crop_position="random")(zero_grid_item)
        _ = Crop(crop_size=(10, 10, 21), crop_position="random")(zero_grid_item)


def test_crop_transform_valid_central_crop_position_check_values(random_grid_item):
    random_grid_item = deepcopy(random_grid_item)
    random_grid_item.subject = np.max(random_grid_item.subject, axis=0)
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(random_grid_item)

    check_elements_not_changed_by_crop(result, random_grid_item)
    assert np.equal(result.input, random_grid_item.input[:, 5:15, 5:15, 5:15]).all()
    assert np.equal(result.field, random_grid_item.field[:, :, :, :, 5:15, 5:15, 5:15]).all()
    assert np.equal(result.subject, random_grid_item.subject[5:15, 5:15, 5:15]).all()
    assert np.equal(result.coils, random_grid_item.coils[:, 5:15, 5:15, 5:15]).all()
    assert np.equal(result.positions, random_grid_item.positions[:, 5:15, 5:15, 5:15]).all()


def test_crop_transform_valid_random_crop_position(zero_grid_item):
    """
    As a test array we take zeros array, so the cropped array would be also zeros array
    """
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    crop = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = crop(zero_grid_item)

    check_elements_not_changed_by_crop(result, zero_grid_item)
    assert np.equal(result.field, zero_grid_item.field[:, :, :, :, 0:10, 0:10, 0:10]).all()
    assert np.equal(result.input, zero_grid_item.input[:, 0:10, 0:10, 0:10]).all()
    assert np.equal(result.subject, zero_grid_item.subject[0:10, 0:10, 0:10]).all()
    assert np.equal(result.coils, zero_grid_item.coils[:, 0:10, 0:10, 0:10]).all()
    assert np.equal(result.positions, zero_grid_item.positions[:, 0:10, 0:10, 0:10]).all()


def test_crop_transform_actions_not_inplace(zero_grid_item):
    crop = Crop(crop_size=(10, 10, 10))
    result = crop(zero_grid_item)

    assert result is not zero_grid_item


def test_crop_transform_check_invalid_simulation():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(10, 10, 10))(None)


def test_default_transform_invalid_dataitem():
    with pytest.raises(ValueError):
        _ = DefaultTransform()(None)


def test_default_transform_actions_not_inplace_for_grid(zero_grid_item):
    trans = DefaultTransform()
    result = trans(zero_grid_item)

    assert result is not zero_grid_item


def test_default_transform_actions_not_inplace_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = DefaultTransform()(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_default_transform_check_datatypes(zero_grid_item):
    result = DefaultTransform()(zero_grid_item)
    check_items_datatypes(result, zero_grid_item)


def test_default_transform_check_shapes_for_grid(zero_grid_item):
    result = DefaultTransform()(zero_grid_item)
    check_default_transform_resulting_shapes(result, zero_grid_item)


def test_default_transform_check_shapes_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = DefaultTransform()(random_pointcloud_item)
    check_default_transform_resulting_shapes(result, random_pointcloud_item)


def test_default_transform_check_values_for_grid(random_grid_item):
    result = DefaultTransform()(random_grid_item)
    check_default_transform_resulting_values(result, random_grid_item)


def test_default_transform_check_values_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = DefaultTransform()(random_pointcloud_item)
    check_default_transform_resulting_values(result, random_pointcloud_item)


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


def test_phase_shift_transform_check_invalid_simulation_for_uniform():
    with pytest.raises(ValueError):
        _ = PhaseShift(num_coils=8, sampling_method="uniform")(None)


def test_phase_shift_transform_check_invalid_simulation_for_binomial():
    with pytest.raises(ValueError):
        _ = PhaseShift(num_coils=8, sampling_method="binomial")(None)


def test_phase_shift_transform_check_valid_processing_dtypes_uniform_for_grid(random_grid_item):
    aug = PhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_grid_item)

    check_items_datatypes(result, random_grid_item)


def test_phase_shift_transform_check_valid_processing_dtypes_binomial_for_grid(random_grid_item):
    """
    Indeed the data item does not have a phase property in the normal case before entering the
    phase shifter, but we have anyway created it in the item fixture for easy check later
    """
    aug = PhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_items_datatypes(result, random_grid_item)


def test_phase_shift_transform_check_valid_processing_dtypes_uniform_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_phase_shift_transform_check_valid_processing_dtypes_binomial_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_phase_shift_transform_check_valid_processing_shapes_uniform_for_grid(random_grid_item):
    aug = PhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 20, 20, 20) -> (2, 2, 3, 20, 20, 20)
    expected_field_shape = (random_grid_item.field.shape[0], random_grid_item.field.shape[1], *random_grid_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_phase_shift_transform_check_valid_processing_shapes_uniform_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 8000) -> (2, 2, 3, 8000)
    expected_field_shape = (random_pointcloud_item.field.shape[0], random_pointcloud_item.field.shape[1], *random_pointcloud_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    # Coils: (8, 8000) -> (2, 8000)
    assert result.coils.shape == (2, *random_pointcloud_item.coils.shape[1:])


def test_phase_shift_transform_check_valid_processing_shapes_binomial_for_grid(random_grid_item):
    aug = PhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 20, 20, 20) -> (2, 2, 3, 20, 20, 20)
    expected_field_shape = (random_grid_item.field.shape[0], random_grid_item.field.shape[1], *random_grid_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_phase_shift_transform_check_valid_processing_shapes_binomial_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 8000) -> (2, 2, 3, 8000)
    expected_field_shape = (random_pointcloud_item.field.shape[0], random_pointcloud_item.field.shape[1], *random_pointcloud_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    # Coils: (8, 8000) -> (2, 8000)
    assert result.coils.shape == (2, *random_pointcloud_item.coils.shape[1:])


def test_phase_shift_transform_check_values_uniform_for_grid(random_grid_item):
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_grid_item)

    check_constant_values_not_changed_by_phase_shift(result, random_grid_item)
    check_complex_number_calculations_in_phase_shift(result, random_grid_item)


def test_phase_shift_transform_check_values_uniform_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_phase_shift(result, random_pointcloud_item)


def test_phase_shift_transform_check_values_binomial_for_grid(random_grid_item):
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_grid_item)

    check_constant_values_not_changed_by_phase_shift(result, random_grid_item)
    check_complex_number_calculations_in_phase_shift(result, random_grid_item)


def test_phase_shift_transform_check_values_binomial_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_phase_shift(result, random_pointcloud_item)


def test_phase_shift_transform_check_not_inplace_processing_for_grid_uniform(random_grid_item):
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_grid_item)

    assert result is not random_grid_item


def test_phase_shift_transform_check_not_inplace_processing_for_grid_binomial(random_grid_item):
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_grid_item)

    assert result is not random_grid_item


def test_phase_shift_transform_check_not_inplace_processing_for_pointcloud_uniform(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_phase_shift_transform_check_not_inplace_processing_for_pointcloud_binomial(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_point_sampling_transform_check_points_sampling_param_is_saved_int():
    aug = PointSampling(points_sampled=1)
    assert type(aug.points_sampled) == int
    assert aug.points_sampled == 1


def test_point_sampling_transform_check_points_sampling_param_is_saved_float():
    aug = PointSampling(points_sampled=0.5)
    assert type(aug.points_sampled) == float
    assert aug.points_sampled == 0.5


def test_point_sampling_transform_check_points_sampling_param_invalid():
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled="value")


def test_point_sampling_transform_check_invalid_simulation():
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=1)(None)

    
def test_point_sampling_transform_check_points_sampling_integer_equal_zero(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=0)(random_pointcloud_item)


def test_point_sampling_transform_check_points_sampling_float_equal_zero(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=0.0)(random_pointcloud_item)


def test_point_sampling_transform_check_points_sampling_integer_less_than_zero(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=-1)(random_pointcloud_item)


def test_point_sampling_transform_check_points_sampling_float_less_than_zero(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=-1.0)(random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_bigger_than_points_in_total(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=8001)(random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_equal_to_points_in_total(random_pointcloud_item):
        random_pointcloud_item = deepcopy(random_pointcloud_item)
        random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
        result = PointSampling(points_sampled=8000)(random_pointcloud_item)

        check_items_shapes_supposed_to_be_equal(result, random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_less_than_points_in_total(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=4000)(random_pointcloud_item)

    assert result.input.shape == (3, 4000)
    assert result.field.shape == (2, 2, 8, 3, 4000)
    assert result.subject.shape == (4000,)
    assert result.positions.shape == (3, 4000)
    assert result.coils.shape == (8, 4000)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_equal_to_points_in_total(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=1.0)(random_pointcloud_item)

    check_items_shapes_supposed_to_be_equal(result, random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_less_than_points_in_total(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=0.5)(random_pointcloud_item)

    assert result.input.shape == (3, 4000)
    assert result.field.shape == (2, 2, 8, 3, 4000)
    assert result.subject.shape == (4000,)
    assert result.positions.shape == (3, 4000)
    assert result.coils.shape == (8, 4000)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_bigger_than_points_in_total(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=1.0001)(random_pointcloud_item)


def test_points_sampling_transform_check_not_inplace_processing_for_float(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=0.5)(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_points_sampling_transform_check_not_inplace_processing_for_int(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=4000)(random_pointcloud_item)

    assert result is not random_pointcloud_item


def check_grid_phase_shift_transform_check_parent():
    assert issubclass(GridPhaseShift, PhaseShift)


def check_point_phase_shift_transform_check_parent():
    assert issubclass(PointPhaseShift, PhaseShift)


def test_grid_phase_shift_transform_check_properties_uniform():
    aug = GridPhaseShift(num_coils=8, sampling_method="uniform")

    assert aug.num_coils == 8
    assert aug.sampling_method == "uniform"


def test_grid_phase_shift_transform_check_properties_binomial():
    aug = GridPhaseShift(num_coils=8, sampling_method="binomial")

    assert aug.num_coils == 8
    assert aug.sampling_method == "binomial"


def test_grid_phase_shift_transform_check_properties_invalid_sampling_method():
    with pytest.raises(ValueError):
        _ = GridPhaseShift(num_coils=8, sampling_method="invalid")


def test_grid_phase_shift_transform_check_invalid_simulation_for_uniform():
    with pytest.raises(ValueError):
        _ = GridPhaseShift(num_coils=8, sampling_method="uniform")(None)


def test_grid_phase_shift_transform_check_invalid_simulation_for_binomial():
    with pytest.raises(ValueError):
        _ = PhaseShift(num_coils=8, sampling_method="binomial")(None)


def test_grid_phase_shift_transform_check_valid_processing_dtypes_uniform(random_grid_item):
    aug = GridPhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_grid_item)

    check_items_datatypes(result, random_grid_item)


def test_grid_phase_shift_transform_check_valid_processing_dtypes_binomial(random_grid_item):
    """
    Indeed the data item does not have a phase property in the normal case before entering the
    phase shifter, but we have anyway created it in the item fixture for easy check later
    """
    aug = GridPhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_items_datatypes(result, random_grid_item)


def test_grid_phase_shift_transform_check_valid_processing_shapes_uniform(random_grid_item):
    aug = GridPhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 20, 20, 20) -> (2, 2, 3, 20, 20, 20)
    expected_field_shape = (random_grid_item.field.shape[0], random_grid_item.field.shape[1], *random_grid_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_grid_phase_shift_transform_check_valid_processing_shapes_binomial(random_grid_item):
    aug = GridPhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 20, 20, 20) -> (2, 2, 3, 20, 20, 20)
    expected_field_shape = (random_grid_item.field.shape[0], random_grid_item.field.shape[1], *random_grid_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_grid_phase_shift_transform_check_values_uniform(random_grid_item):
    result = GridPhaseShift(num_coils=8, sampling_method="uniform")(random_grid_item)

    check_constant_values_not_changed_by_phase_shift(result, random_grid_item)
    check_complex_number_calculations_in_phase_shift(result, random_grid_item)


def test_grid_phase_shift_transform_check_values_binomial(random_grid_item):
    result = GridPhaseShift(num_coils=8, sampling_method="binomial")(random_grid_item)

    check_constant_values_not_changed_by_phase_shift(result, random_grid_item)
    check_complex_number_calculations_in_phase_shift(result, random_grid_item)


def test_grid_phase_shift_transform_check_not_inplace_processing_for_uniform(random_grid_item):
    result = GridPhaseShift(num_coils=8, sampling_method="uniform")(random_grid_item)

    assert result is not random_grid_item


def test_grid_phase_shift_transform_check_not_inplace_processing_for_binomial(random_grid_item):
    result = GridPhaseShift(num_coils=8, sampling_method="binomial")(random_grid_item)

    assert result is not random_grid_item


def test_point_phase_shift_transform_check_properties_uniform():
    aug = PointPhaseShift(num_coils=8, sampling_method="uniform")

    assert aug.num_coils == 8
    assert aug.sampling_method == "uniform"


def test_point_phase_shift_transform_check_properties_binomial():
    aug = PointPhaseShift(num_coils=8, sampling_method="binomial")

    assert aug.num_coils == 8
    assert aug.sampling_method == "binomial"


def test_point_phase_shift_transform_check_properties_invalid_sampling_method():
    with pytest.raises(ValueError):
        _ = PointPhaseShift(num_coils=8, sampling_method="invalid")


def test_point_phase_shift_transform_check_invalid_simulation_for_uniform():
    with pytest.raises(ValueError):
        _ = PointPhaseShift(num_coils=8, sampling_method="uniform")(None)


def test_point_phase_shift_transform_check_invalid_simulation_for_binomial():
    with pytest.raises(ValueError):
        _ = PointPhaseShift(num_coils=8, sampling_method="binomial")(None)


def test_point_phase_shift_transform_check_valid_processing_dtypes_uniform(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_valid_processing_dtypes_binomial(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_valid_processing_shapes_uniform(random_pointcloud_item):
    """
    Validates shapes after PointPhaseShift transform with uniform sampling.
    """
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    expected_field_shape = (random_pointcloud_item.field.shape[0], random_pointcloud_item.field.shape[1], *random_pointcloud_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    assert result.coils.shape == (2, *random_pointcloud_item.coils.shape[1:])


def test_point_phase_shift_transform_check_valid_processing_shapes_binomial(random_pointcloud_item):
    """
    Validates shapes after PointPhaseShift transform with binomial sampling.
    """
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    expected_field_shape = (random_pointcloud_item.field.shape[0], random_pointcloud_item.field.shape[1], *random_pointcloud_item.field.shape[3:])
    assert result.field.shape == expected_field_shape
    assert result.coils.shape == tuple([2] + list(random_pointcloud_item.coils.shape[1:]))


def test_point_phase_shift_transform_check_values_uniform(random_pointcloud_item):
    """
    Validates computed values after PointPhaseShift transform with uniform sampling.
    Checks that phase shift calculations are correct for the new consistent shape format.
    """
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_pointscloud_phase_shift(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_values_binomial(random_pointcloud_item):
    """
    Validates computed values after PointPhaseShift transform with binomial sampling.
    Checks that phase shift calculations are correct for the new consistent shape format.
    """
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_pointscloud_phase_shift(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_not_inplace_processing_for_uniform(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_point_phase_shift_transform_check_not_inplace_processing_for_binomial(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_rotate_transform_invalid_rot_angle():
    with pytest.raises(ValueError):
        _ = Rotate(rot_angle="invalid")


def test_rotate_transform_invalid_rot_axis():
    with pytest.raises(ValueError):
        _ = Rotate(rot_axis="invalid")


def test_rotate_transform_check_properties_random_z():
    aug = Rotate(rot_angle="random", rot_axis="z")
    
    assert aug.rot_angle == "random"
    assert aug.rot_plane == (-3, -2)


def test_rotate_transform_check_properties_90_z():
    aug = Rotate(rot_angle="90", rot_axis="z")
    
    assert aug.rot_angle == "90"
    assert aug.n_rot == 0
    assert aug.rot_plane == (-3, -2)


def test_rotate_transform_check_properties_random_x():
    aug = Rotate(rot_angle="random", rot_axis="x")
    
    assert aug.rot_angle == "random"
    assert aug.rot_plane == (-2, -1)


def test_rotate_transform_check_properties_random_y():
    aug = Rotate(rot_angle="random", rot_axis="y")
    
    assert aug.rot_angle == "random"
    assert aug.rot_plane == (-3, -1)


def test_rotate_transform_check_datatypes_for_grid(random_grid_item):
    aug = Rotate(rot_angle="90", rot_axis="z")
    result = aug(random_grid_item)
    
    check_items_datatypes(result, random_grid_item)


def test_rotate_transform_check_shapes_for_grid(random_grid_item):
    aug = Rotate(rot_angle="90", rot_axis="z")
    result = aug(random_grid_item)
    
    check_items_shapes_supposed_to_be_equal(result, random_grid_item)


def test_rotate_transform_90_degrees_z_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Rotate(rot_angle="90", rot_axis="z")
    result = aug(zero_grid_item)
    
    assert result.simulation == zero_grid_item.simulation
    assert np.equal(result.phase, zero_grid_item.phase).all()
    assert np.equal(result.mask, zero_grid_item.mask).all()
    assert result.dtype == zero_grid_item.dtype
    assert np.equal(result.truncation_coefficients, zero_grid_item.truncation_coefficients).all()
    
    assert np.equal(result.input, np.rot90(zero_grid_item.input, k=1, axes=(-3, -2))).all()
    assert np.equal(result.field, np.rot90(zero_grid_item.field, k=1, axes=(-3, -2))).all()
    assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=1, axes=(-3, -2))).all()
    assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=1, axes=(-3, -2))).all()
    assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=1, axes=(-3, -2))).all()


def test_rotate_transform_90_degrees_x_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Rotate(rot_angle="90", rot_axis="x")
    result = aug(zero_grid_item)
    
    assert np.equal(result.input, np.rot90(zero_grid_item.input, k=1, axes=(-2, -1))).all()
    assert np.equal(result.field, np.rot90(zero_grid_item.field, k=1, axes=(-2, -1))).all()
    assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=1, axes=(-2, -1))).all()
    assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=1, axes=(-2, -1))).all()
    assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=1, axes=(-2, -1))).all()


def test_rotate_transform_90_degrees_y_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Rotate(rot_angle="90", rot_axis="y")
    result = aug(zero_grid_item)
    
    assert np.equal(result.input, np.rot90(zero_grid_item.input, k=1, axes=(-3, -1))).all()
    assert np.equal(result.field, np.rot90(zero_grid_item.field, k=1, axes=(-3, -1))).all()
    assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=1, axes=(-3, -1))).all()
    assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=1, axes=(-3, -1))).all()
    assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=1, axes=(-3, -1))).all()


def test_rotate_transform_random_angle_z_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Rotate(rot_angle="random", rot_axis="z")
    result = aug(zero_grid_item)
    
    assert aug.n_rot in [0, 1, 2, 3]
    assert np.equal(result.input, np.rot90(zero_grid_item.input, k=aug.n_rot, axes=(-3, -2))).all()
    assert np.equal(result.field, np.rot90(zero_grid_item.field, k=aug.n_rot, axes=(-3, -2))).all()
    assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=aug.n_rot, axes=(-3, -2))).all()
    assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=aug.n_rot, axes=(-3, -2))).all()
    assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=aug.n_rot, axes=(-3, -2))).all()


def test_rotate_transform_random_angle_correct_range_parameters(zero_grid_item):
    """
    Validates that np.random.randint is called with correct parameters (0, 4)
    when rot_angle='random' to ensure all four rotation values are possible.
    """
    with patch('numpy.random.randint', return_value=2) as mock_randint:
        aug = Rotate(rot_angle="random", rot_axis="z")
        _ = aug(zero_grid_item)
        
        mock_randint.assert_called_once_with(0, 4)
        assert aug.n_rot == 2


def test_rotate_transform_random_angle_k0_rotation(zero_grid_item):
    """
    Validates that rotation with k=0 (0 degrees) is correctly applied when sampled.
    """
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    
    with patch('numpy.random.randint', return_value=0):
        aug = Rotate(rot_angle="random", rot_axis="z")
        result = aug(zero_grid_item)
        
        assert aug.n_rot == 0
        assert np.equal(result.input, zero_grid_item.input).all()
        assert np.equal(result.field, zero_grid_item.field).all()
        assert np.equal(result.subject, zero_grid_item.subject).all()
        assert np.equal(result.coils, zero_grid_item.coils).all()
        assert np.equal(result.positions, zero_grid_item.positions).all()


def test_rotate_transform_random_angle_k1_rotation(zero_grid_item):
    """
    Validates that rotation with k=1 (90 degrees) is correctly applied when sampled.
    """
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    
    with patch('numpy.random.randint', return_value=1):
        aug = Rotate(rot_angle="random", rot_axis="z")
        result = aug(zero_grid_item)
        
        assert aug.n_rot == 1
        assert np.equal(result.input, np.rot90(zero_grid_item.input, k=1, axes=(-3, -2))).all()
        assert np.equal(result.field, np.rot90(zero_grid_item.field, k=1, axes=(-3, -2))).all()
        assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=1, axes=(-3, -2))).all()
        assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=1, axes=(-3, -2))).all()
        assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=1, axes=(-3, -2))).all()


def test_rotate_transform_random_angle_k2_rotation(zero_grid_item):
    """
    Validates that rotation with k=2 (180 degrees) is correctly applied when sampled.
    """
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    
    with patch('numpy.random.randint', return_value=2):
        aug = Rotate(rot_angle="random", rot_axis="z")
        result = aug(zero_grid_item)
        
        assert aug.n_rot == 2
        assert np.equal(result.input, np.rot90(zero_grid_item.input, k=2, axes=(-3, -2))).all()
        assert np.equal(result.field, np.rot90(zero_grid_item.field, k=2, axes=(-3, -2))).all()
        assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=2, axes=(-3, -2))).all()
        assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=2, axes=(-3, -2))).all()
        assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=2, axes=(-3, -2))).all()


def test_rotate_transform_random_angle_k3_rotation(zero_grid_item):
    """
    Validates that rotation with k=3 (270 degrees) is correctly applied when sampled.
    """
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    
    with patch('numpy.random.randint', return_value=3):
        aug = Rotate(rot_angle="random", rot_axis="z")
        result = aug(zero_grid_item)
        
        assert aug.n_rot == 3
        assert np.equal(result.input, np.rot90(zero_grid_item.input, k=3, axes=(-3, -2))).all()
        assert np.equal(result.field, np.rot90(zero_grid_item.field, k=3, axes=(-3, -2))).all()
        assert np.equal(result.subject, np.rot90(zero_grid_item.subject, k=3, axes=(-3, -2))).all()
        assert np.equal(result.coils, np.rot90(zero_grid_item.coils, k=3, axes=(-3, -2))).all()
        assert np.equal(result.positions, np.rot90(zero_grid_item.positions, k=3, axes=(-3, -2))).all()


def test_rotate_transform_not_inplace_processing_for_grid(random_grid_item):
    aug = Rotate(rot_angle="90", rot_axis="z")
    result = aug(random_grid_item)
    
    assert result is not random_grid_item


def test_mirror_transform_invalid_mirror_axis():
    with pytest.raises(ValueError):
        _ = Mirror(mirror_axis="invalid")


def test_mirror_transform_invalid_mirror_prob_type():
    with pytest.raises(ValueError):
        _ = Mirror(mirror_prob="invalid")


def test_mirror_transform_invalid_mirror_prob_negative():
    with pytest.raises(ValueError):
        _ = Mirror(mirror_prob=-0.5)


def test_mirror_transform_invalid_mirror_prob_greater_than_one():
    with pytest.raises(ValueError):
        _ = Mirror(mirror_prob=1.5)


def test_mirror_transform_check_properties_z_axis():
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    
    assert aug.mirror_axis == -1
    assert aug.mirror_prob == 1.0


def test_mirror_transform_check_properties_x_axis():
    aug = Mirror(mirror_axis="x", mirror_prob=0.5)
    
    assert aug.mirror_axis == -3
    assert aug.mirror_prob == 0.5


def test_mirror_transform_check_properties_y_axis():
    aug = Mirror(mirror_axis="y", mirror_prob=0.8)
    
    assert aug.mirror_axis == -2
    assert aug.mirror_prob == 0.8


def test_mirror_transform_check_datatypes_for_grid(random_grid_item):
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_grid_item)
    
    check_items_datatypes(result, random_grid_item)


def test_mirror_transform_check_datatypes_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_pointcloud_item)
    
    check_items_datatypes(result, random_pointcloud_item)


def test_mirror_transform_check_shapes_for_grid(random_grid_item):
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_grid_item)
    
    check_items_shapes_supposed_to_be_equal(result, random_grid_item)


def test_mirror_transform_check_shapes_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_pointcloud_item)
    
    check_items_shapes_supposed_to_be_equal(result, random_pointcloud_item)


def test_mirror_transform_z_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(zero_grid_item)
    
    assert result.simulation == zero_grid_item.simulation
    assert np.equal(result.phase, zero_grid_item.phase).all()
    assert np.equal(result.mask, zero_grid_item.mask).all()
    assert result.dtype == zero_grid_item.dtype
    assert np.equal(result.truncation_coefficients, zero_grid_item.truncation_coefficients).all()
    
    assert np.equal(result.input, np.flip(zero_grid_item.input, axis=-1)).all()
    assert np.equal(result.field, np.flip(zero_grid_item.field, axis=-1)).all()
    assert np.equal(result.subject, np.flip(zero_grid_item.subject, axis=-1)).all()
    assert np.equal(result.coils, np.flip(zero_grid_item.coils, axis=-1)).all()
    assert np.equal(result.positions, np.flip(zero_grid_item.positions, axis=-1)).all()


def test_mirror_transform_x_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Mirror(mirror_axis="x", mirror_prob=1.0)
    result = aug(zero_grid_item)
    
    assert np.equal(result.input, np.flip(zero_grid_item.input, axis=-3)).all()
    assert np.equal(result.field, np.flip(zero_grid_item.field, axis=-3)).all()
    assert np.equal(result.subject, np.flip(zero_grid_item.subject, axis=-3)).all()
    assert np.equal(result.coils, np.flip(zero_grid_item.coils, axis=-3)).all()
    assert np.equal(result.positions, np.flip(zero_grid_item.positions, axis=-3)).all()


def test_mirror_transform_y_axis_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Mirror(mirror_axis="y", mirror_prob=1.0)
    result = aug(zero_grid_item)
    
    assert np.equal(result.input, np.flip(zero_grid_item.input, axis=-2)).all()
    assert np.equal(result.field, np.flip(zero_grid_item.field, axis=-2)).all()
    assert np.equal(result.subject, np.flip(zero_grid_item.subject, axis=-2)).all()
    assert np.equal(result.coils, np.flip(zero_grid_item.coils, axis=-2)).all()
    assert np.equal(result.positions, np.flip(zero_grid_item.positions, axis=-2)).all()


def test_mirror_transform_probability_zero_for_grid(zero_grid_item):
    zero_grid_item = deepcopy(zero_grid_item)
    zero_grid_item.subject = np.max(zero_grid_item.subject, axis=0)
    aug = Mirror(mirror_axis="z", mirror_prob=0.0)
    result = aug(zero_grid_item)
    
    assert np.equal(result.input, zero_grid_item.input).all()
    assert np.equal(result.field, zero_grid_item.field).all()
    assert np.equal(result.subject, zero_grid_item.subject).all()
    assert np.equal(result.coils, zero_grid_item.coils).all()
    assert np.equal(result.positions, zero_grid_item.positions).all()


def test_mirror_transform_z_axis_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_pointcloud_item)
    
    assert result.simulation == random_pointcloud_item.simulation
    assert np.equal(result.phase, random_pointcloud_item.phase).all()
    assert np.equal(result.mask, random_pointcloud_item.mask).all()
    assert result.dtype == random_pointcloud_item.dtype
    assert np.equal(result.truncation_coefficients, random_pointcloud_item.truncation_coefficients).all()


def test_mirror_transform_not_inplace_processing_for_grid(random_grid_item):
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_grid_item)
    
    assert result is not random_grid_item


def test_mirror_transform_not_inplace_processing_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    aug = Mirror(mirror_axis="z", mirror_prob=1.0)
    result = aug(random_pointcloud_item)
    
    assert result is not random_pointcloud_item
