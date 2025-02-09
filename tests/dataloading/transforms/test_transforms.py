import pytest

import numpy as np
from einops import rearrange

from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.data.transforms import (
    Compose, Crop, GridPhaseShift, PointPhaseShift, PointSampling, 
    PhaseShift, BaseTransform, DefaultTransform, PointFeatureRearrange
)
from tests.dataloading.transforms.helpers import (
    FirstAugmentation, SecondAugmentation, ThirdAugmentation, check_items_datatypes,
    check_cropped_shapes, check_items_shapes_suppsed_to_be_equal, check_elements_not_changed_by_crop,
    check_constant_shapes_not_changed_except_for_field_coils, check_constant_values_not_changed_by_phase_shift,
    check_default_transform_resulting_shapes, check_default_transform_resulting_values,
    check_complex_number_calculations_in_phase_shift, check_complex_number_calculations_in_pointscloud_phase_shift,
    check_pointcloud_feature_rearrange_shapes_field_coils, check_constant_values_not_changed_except_for_field_coils,
    check_pointcloud_feature_rearrange_values_field_coils
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

    assert result_item.simulation == "123"


def test_compose_running_order_for_pointcloud(random_pointcloud_item):
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])
    result_item = aug(random_pointcloud_item)

    assert result_item.simulation == "123"


def test_compose_transform_not_inplace_processing_for_grid(zero_grid_item):
    aug = Compose([FirstAugmentation(), SecondAugmentation(), ThirdAugmentation()])
    result_item = aug(zero_grid_item)

    assert result_item is not zero_grid_item


def test_compose_transform_not_inplace_processing_for_pointcloud(random_pointcloud_item):
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
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(zero_grid_item)
    check_cropped_shapes(result)


def test_crop_transform_valid_random_crop_position_shape(zero_grid_item):
    augment = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = augment(zero_grid_item)
    check_cropped_shapes(result)


def test_crop_transform_crop_size_matches_original_central_crop_position(zero_grid_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="center")
    result = crop(zero_grid_item)
    check_items_shapes_suppsed_to_be_equal(result, zero_grid_item)


def test_crop_transform_crop_size_matches_original_random_crop_position(zero_grid_item):
    crop = Crop(crop_size=(20, 20, 20), crop_position="random")
    result = crop(zero_grid_item)
    check_items_shapes_suppsed_to_be_equal(result, zero_grid_item)


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
    augment = Crop(crop_size=(10, 10, 10), crop_position="center")
    result = augment(random_grid_item)

    check_elements_not_changed_by_crop(result, random_grid_item)
    assert np.equal(result.input, random_grid_item.input[:, 5:15, 5:15, 5:15]).all()
    assert np.equal(result.field, random_grid_item.field[:, :, :, 5:15, 5:15, 5:15, :]).all()
    assert np.equal(result.subject, random_grid_item.subject[5:15, 5:15, 5:15]).all()
    assert np.equal(result.coils, random_grid_item.coils[5:15, 5:15, 5:15, :]).all()


def test_crop_transform_valid_random_crop_position(zero_grid_item):
    """
    As a test array we take zeros array, so the cropped array would be also zeros array
    """
    crop = Crop(crop_size=(10, 10, 10), crop_position="random")
    result = crop(zero_grid_item)

    check_elements_not_changed_by_crop(result, zero_grid_item)
    assert np.equal(result.field, zero_grid_item.field[:, :, :, 0:10, 0:10, 0:10, :]).all()
    assert np.equal(result.input, zero_grid_item.input[:, 0:10, 0:10, 0:10]).all()
    assert np.equal(result.subject, zero_grid_item.subject[0:10, 0:10, 0:10]).all()
    assert np.equal(result.coils, zero_grid_item.coils[0:10, 0:10, 0:10, :]).all()


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
    result = DefaultTransform()(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_default_transform_check_datatypes(zero_grid_item):
    result = DefaultTransform()(zero_grid_item)
    check_items_datatypes(result, zero_grid_item)


def test_default_transform_check_shapes_for_grid(zero_grid_item):
    result = DefaultTransform()(zero_grid_item)
    check_default_transform_resulting_shapes(result, zero_grid_item)


def test_default_transform_check_shapes_for_pointcloud(random_pointcloud_item):
    result = DefaultTransform()(random_pointcloud_item)
    check_default_transform_resulting_shapes(result, random_pointcloud_item)


def test_default_transform_check_values_for_grid(random_grid_item):
    result = DefaultTransform()(random_grid_item)
    check_default_transform_resulting_values(result, random_grid_item)


def test_default_transform_check_values_for_pointcloud(random_pointcloud_item):
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
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_phase_shift_transform_check_valid_processing_dtypes_binomial_for_pointcloud(random_pointcloud_item):
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_phase_shift_transform_check_valid_processing_shapes_uniform_for_grid(random_grid_item):
    aug = PhaseShift(num_coils=8, sampling_method="uniform")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    assert result.field.shape == random_grid_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_grid_item.coils.shape[:-1]))


def test_phase_shift_transform_check_valid_processing_shapes_uniform_for_pointcloud(random_pointcloud_item):
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    assert result.field.shape == random_pointcloud_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_pointcloud_item.coils.shape[:-1]))


def test_phase_shift_transform_check_valid_processing_shapes_binomial_for_grid(random_grid_item):
    aug = PhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    assert result.field.shape == random_grid_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_grid_item.coils.shape[:-1]))


def check_phase_shift_transform_check_valid_processing_shapes_binomial_for_pointcloud(random_pointcloud_item):
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    assert result.field.shape == random_pointcloud_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_pointcloud_item.coils.shape[:-1]))


def test_phase_shift_transform_check_values_uniform_for_grid(random_grid_item):
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_grid_item)

    check_constant_values_not_changed_by_phase_shift(result, random_grid_item)
    check_complex_number_calculations_in_phase_shift(result, random_grid_item)


def test_phase_shift_transform_check_values_uniform_for_pointcloud(random_pointcloud_item):
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_phase_shift(result, random_pointcloud_item)


def test_phase_shift_transform_check_values_binomial_for_grid(random_grid_item):
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_grid_item)

    check_constant_values_not_changed_by_phase_shift(result, random_grid_item)
    check_complex_number_calculations_in_phase_shift(result, random_grid_item)


def test_phase_shift_transform_check_values_binomial_for_pointcloud(random_pointcloud_item):
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
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_phase_shift_transform_check_not_inplace_processing_for_pointcloud_binomial(random_pointcloud_item):
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
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=0)(random_pointcloud_item)


def test_point_sampling_transform_check_points_sampling_float_equal_zero(random_pointcloud_item):
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=0.0)(random_pointcloud_item)


def test_point_sampling_transform_check_points_sampling_integer_less_than_zero(random_pointcloud_item):
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=-1)(random_pointcloud_item)


def test_point_sampling_transform_check_points_sampling_float_less_than_zero(random_pointcloud_item):
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=-1.0)(random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_bigger_than_points_in_total(random_pointcloud_item):
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=8001)(random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_equal_to_points_in_total(random_pointcloud_item):
        result = PointSampling(points_sampled=8000)(random_pointcloud_item)

        check_items_shapes_suppsed_to_be_equal(result, random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_less_than_points_in_total(random_pointcloud_item):
    result = PointSampling(points_sampled=4000)(random_pointcloud_item)

    assert result.input.shape == (4000, 3)
    assert result.field.shape == (2, 2, 4000, 3, 8)
    assert result.subject.shape == (4000,)
    assert result.positions.shape == (4000, 3)
    assert result.coils.shape == (4000, 8)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_equal_to_points_in_total(random_pointcloud_item):
    result = PointSampling(points_sampled=1.0)(random_pointcloud_item)

    check_items_shapes_suppsed_to_be_equal(result, random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_less_than_points_in_total(random_pointcloud_item):
    result = PointSampling(points_sampled=0.5)(random_pointcloud_item)

    assert result.input.shape == (4000, 3)
    assert result.field.shape == (2, 2, 4000, 3, 8)
    assert result.subject.shape == (4000,)
    assert result.positions.shape == (4000, 3)
    assert result.coils.shape == (4000, 8)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_bigger_than_points_in_total(random_pointcloud_item):
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=1.0001)(random_pointcloud_item)


def test_points_sampling_transform_check_not_inplace_processing_for_float(random_pointcloud_item):
    result = PointSampling(points_sampled=0.5)(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_points_sampling_transform_check_not_inplace_processing_for_int(random_pointcloud_item):
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
    assert result.field.shape == random_grid_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_grid_item.coils.shape[:-1]))


def test_grid_phase_shift_transform_check_valid_processing_shapes_binomial(random_grid_item):
    aug = GridPhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    assert result.field.shape == random_grid_item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_grid_item.coils.shape[:-1]))


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
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_valid_processing_dtypes_binomial(random_pointcloud_item):
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)
    check_items_datatypes(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_valid_processing_shapes_uniform(random_pointcloud_item):
    """
    This test assumes the preprocessing did not standartize the axis position and `fieldxyz` and `positions` are having different order
    """
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    real_expected_array = np.ascontiguousarray(
        rearrange(random_pointcloud_item.field, "he reim positions fieldxyz coils -> he reim fieldxyz positions coils")
    ).astype(np.float32)
    assert result.field.shape == real_expected_array.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_pointcloud_item.coils.shape[:-1]))


def test_point_phase_shift_transform_check_valid_processing_shapes_binomial(random_pointcloud_item):
    """
    This test assumes the preprocessing did not standartize the axis position and `fieldxyz` and `positions` are having different order
    """
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    real_expected_array = np.ascontiguousarray(
        rearrange(random_pointcloud_item.field, "he reim positions fieldxyz coils -> he reim fieldxyz positions coils")
    ).astype(np.float32)
    assert result.field.shape == real_expected_array.shape[:-1]
    assert result.coils.shape == tuple([2] + list(random_pointcloud_item.coils.shape[:-1]))


def test_point_phase_shift_transform_check_values_uniform(random_pointcloud_item):
    """
    This test assumes the preprocessing did not standartize the axis position and `fieldxyz` and `positions` are having different order
    """
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_pointscloud_phase_shift(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_values_binomial(random_pointcloud_item):
    """
    This test assumes the preprocessing did not standartize the axis position and `fieldxyz` and `positions` are having different order
    """
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_values_not_changed_by_phase_shift(result, random_pointcloud_item)
    check_complex_number_calculations_in_pointscloud_phase_shift(result, random_pointcloud_item)


def test_point_phase_shift_transform_check_not_inplace_processing_for_uniform(random_pointcloud_item):
    result = PointPhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_point_phase_shift_transform_check_not_inplace_processing_for_binomial(random_pointcloud_item):
    result = PointPhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    assert result is not random_pointcloud_item


def test_point_feature_rearrange_transform_invalid_dataitem():
    with pytest.raises(ValueError):
        _ = PointFeatureRearrange(num_coils=8)(None)


def test_point_feature_rearrange_transform_actions_not_inplace(random_pointcloud_item_for_features_rearrange):
    result = PointFeatureRearrange(num_coils=8)(random_pointcloud_item_for_features_rearrange)

    assert result is not random_pointcloud_item_for_features_rearrange


def test_point_feature_rearrange_transform_check_datatypes(random_pointcloud_item_for_features_rearrange):
    result = PointFeatureRearrange(num_coils=8)(random_pointcloud_item_for_features_rearrange)
    check_items_datatypes(result, random_pointcloud_item_for_features_rearrange)


def test_point_feature_rearrange_transform_check_shapes(random_pointcloud_item_for_features_rearrange):
    result = PointFeatureRearrange(num_coils=8)(random_pointcloud_item_for_features_rearrange)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item_for_features_rearrange)
    check_pointcloud_feature_rearrange_shapes_field_coils(result, random_pointcloud_item_for_features_rearrange)


def test_point_feature_rearrange_transform_check_values(random_pointcloud_item_for_features_rearrange):
    result = PointFeatureRearrange(num_coils=8)(random_pointcloud_item_for_features_rearrange)

    check_constant_values_not_changed_except_for_field_coils(result, random_pointcloud_item_for_features_rearrange)
    check_pointcloud_feature_rearrange_values_field_coils(result, random_pointcloud_item_for_features_rearrange)
