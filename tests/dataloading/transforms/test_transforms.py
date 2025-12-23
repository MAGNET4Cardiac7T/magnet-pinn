from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.data.transforms import (
    BaseTransform,
    CoilEnumeratorPhaseShift,
    Compose,
    Crop,
    DefaultTransform,
    GridPhaseShift,
    Mirror,
    PhaseShift,
    PointPhaseShift,
    PointSampling,
    Rotate,
    check_transforms,
)
from tests.dataloading.transforms.helpers import (
    FirstAugmentation,
    SecondAugmentation,
    ThirdAugmentation,
    check_complex_number_calculations_in_phase_shift,
    check_complex_number_calculations_in_pointscloud_phase_shift,
    check_constant_shapes_not_changed_except_for_field_coils,
    check_constant_values_not_changed_by_phase_shift,
    check_cropped_shapes,
    check_default_transform_resulting_shapes,
    check_default_transform_resulting_values,
    check_elements_not_changed_by_crop,
    check_items_datatypes,
    check_items_shapes_supposed_to_be_equal,
)


def check_base_transform_callable():
    assert callable(BaseTransform)


def test_compose_none_transformations_given():
    with pytest.raises(ValueError):
        _ = Compose(None)  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_compose_empty_list():
    with pytest.raises(ValueError):
        _ = Compose([])


def test_compose_with_none_transformations():
    with pytest.raises(ValueError):
        _ = Compose([None, None])  # type: ignore[list-item]  # Intentionally testing invalid input


def test_compose_with_mixed_none_transformation():
    """Test that Compose raises ValueError when augmentation list contains None mixed with valid transforms."""
    with pytest.raises(ValueError, match="Augmentation can not be None"):
        _ = Compose([DefaultTransform(), None])  # type: ignore[list-item]  # Intentionally testing invalid input


def test_compose_with_invalid_type_transform():
    with pytest.raises(ValueError):
        _ = Compose(["transformation"])  # type: ignore[list-item]  # Intentionally testing invalid input


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
        _ = Crop(crop_size=None)  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_crop_transform_crop_size_invalid_dimensions_number():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(100, 100))  # type: ignore[arg-type]  # Intentionally testing invalid input

    with pytest.raises(ValueError):
        _ = Crop(crop_size=(100, 100, 100, 100))  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_crop_transform_crop_size_invalid_type():
    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, 0, 3.5))  # type: ignore[arg-type]  # Intentionally testing invalid input

    with pytest.raises(ValueError):
        _ = Crop(crop_size=(1, "value", 0))  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_crop_transform_crop_position_invalid_type():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = Crop(crop_size=(1, 1, 1), crop_position="value")  # type: ignore[arg-type]


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
        _ = Crop(crop_size=(10, 10, 10))(None)  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_default_transform_invalid_dataitem():
    with pytest.raises(ValueError):
        _ = DefaultTransform()(None)  # type: ignore[arg-type]  # Intentionally testing invalid input


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
        # Intentionally testing invalid input
        _ = PhaseShift(num_coils=8, sampling_method="invalid")  # type: ignore[arg-type]


def test_phase_shift_transform_check_invalid_simulation_for_uniform():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = PhaseShift(num_coils=8, sampling_method="uniform")(None)  # type: ignore[arg-type]


def test_phase_shift_transform_check_invalid_simulation_for_binomial():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = PhaseShift(num_coils=8, sampling_method="binomial")(None)  # type: ignore[arg-type]


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
    expected_field_shape = (
        random_grid_item.field.shape[0],
        random_grid_item.field.shape[1],
        *random_grid_item.field.shape[3:],
    )
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_phase_shift_transform_check_valid_processing_shapes_uniform_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="uniform")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 8000) -> (2, 2, 3, 8000)
    expected_field_shape = (
        random_pointcloud_item.field.shape[0],
        random_pointcloud_item.field.shape[1],
        *random_pointcloud_item.field.shape[3:],
    )
    assert result.field.shape == expected_field_shape
    # Coils: (8, 8000) -> (2, 8000)
    assert result.coils.shape == (2, *random_pointcloud_item.coils.shape[1:])


def test_phase_shift_transform_check_valid_processing_shapes_binomial_for_grid(random_grid_item):
    aug = PhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 20, 20, 20) -> (2, 2, 3, 20, 20, 20)
    expected_field_shape = (
        random_grid_item.field.shape[0],
        random_grid_item.field.shape[1],
        *random_grid_item.field.shape[3:],
    )
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_phase_shift_transform_check_valid_processing_shapes_binomial_for_pointcloud(random_pointcloud_item):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PhaseShift(num_coils=8, sampling_method="binomial")(random_pointcloud_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_pointcloud_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 8000) -> (2, 2, 3, 8000)
    expected_field_shape = (
        random_pointcloud_item.field.shape[0],
        random_pointcloud_item.field.shape[1],
        *random_pointcloud_item.field.shape[3:],
    )
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
    # Verify exact type is int, not float (isinstance would be incorrect here)
    assert type(aug.points_sampled) == int  # noqa: E721
    assert aug.points_sampled == 1


def test_point_sampling_transform_check_points_sampling_param_is_saved_float():
    aug = PointSampling(points_sampled=0.5)
    # Verify exact type is float, not int (isinstance would be incorrect here)
    assert type(aug.points_sampled) == float  # noqa: E721
    assert aug.points_sampled == 0.5


def test_point_sampling_transform_check_points_sampling_param_invalid():
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled="value")  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_point_sampling_transform_check_invalid_simulation():
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=1)(None)  # type: ignore[arg-type]  # Intentionally testing invalid input


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


def test_points_sampling_transform_check_points_sampling_parameter_int_and_bigger_than_points_in_total(
    random_pointcloud_item,
):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    with pytest.raises(ValueError):
        _ = PointSampling(points_sampled=8001)(random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_equal_to_points_in_total(
    random_pointcloud_item,
):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=8000)(random_pointcloud_item)

    check_items_shapes_supposed_to_be_equal(result, random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_int_and_less_than_points_in_total(
    random_pointcloud_item,
):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=4000)(random_pointcloud_item)

    assert result.input.shape == (3, 4000)
    assert result.field.shape == (2, 2, 8, 3, 4000)
    assert result.subject.shape == (4000,)
    assert result.positions.shape == (3, 4000)
    assert result.coils.shape == (8, 4000)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_equal_to_points_in_total(
    random_pointcloud_item,
):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=1.0)(random_pointcloud_item)

    check_items_shapes_supposed_to_be_equal(result, random_pointcloud_item)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_less_than_points_in_total(
    random_pointcloud_item,
):
    random_pointcloud_item = deepcopy(random_pointcloud_item)
    random_pointcloud_item.subject = np.max(random_pointcloud_item.subject, axis=0)
    result = PointSampling(points_sampled=0.5)(random_pointcloud_item)

    assert result.input.shape == (3, 4000)
    assert result.field.shape == (2, 2, 8, 3, 4000)
    assert result.subject.shape == (4000,)
    assert result.positions.shape == (3, 4000)
    assert result.coils.shape == (8, 4000)


def test_points_sampling_transform_check_points_sampling_parameter_float_and_bigger_than_points_in_total(
    random_pointcloud_item,
):
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
        # Intentionally testing invalid input
        _ = GridPhaseShift(num_coils=8, sampling_method="invalid")  # type: ignore[arg-type]


def test_grid_phase_shift_transform_check_invalid_simulation_for_uniform():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = GridPhaseShift(num_coils=8, sampling_method="uniform")(None)  # type: ignore[arg-type]


def test_grid_phase_shift_transform_check_invalid_simulation_for_binomial():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = PhaseShift(num_coils=8, sampling_method="binomial")(None)  # type: ignore[arg-type]


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
    expected_field_shape = (
        random_grid_item.field.shape[0],
        random_grid_item.field.shape[1],
        *random_grid_item.field.shape[3:],
    )
    assert result.field.shape == expected_field_shape
    # Coils: (8, 20, 20, 20) -> (2, 20, 20, 20)
    assert result.coils.shape == (2, *random_grid_item.coils.shape[1:])


def test_grid_phase_shift_transform_check_valid_processing_shapes_binomial(random_grid_item):
    aug = GridPhaseShift(num_coils=8, sampling_method="binomial")
    result = aug(random_grid_item)

    check_constant_shapes_not_changed_except_for_field_coils(result, random_grid_item)
    # Field: coils dimension (axis 2) removed -> (2, 2, 8, 3, 20, 20, 20) -> (2, 2, 3, 20, 20, 20)
    expected_field_shape = (
        random_grid_item.field.shape[0],
        random_grid_item.field.shape[1],
        *random_grid_item.field.shape[3:],
    )
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
        # Intentionally testing invalid input
        _ = PointPhaseShift(num_coils=8, sampling_method="invalid")  # type: ignore[arg-type]


def test_point_phase_shift_transform_check_invalid_simulation_for_uniform():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = PointPhaseShift(num_coils=8, sampling_method="uniform")(None)  # type: ignore[arg-type]


def test_point_phase_shift_transform_check_invalid_simulation_for_binomial():
    with pytest.raises(ValueError):
        # Intentionally testing invalid input
        _ = PointPhaseShift(num_coils=8, sampling_method="binomial")(None)  # type: ignore[arg-type]


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
    expected_field_shape = (
        random_pointcloud_item.field.shape[0],
        random_pointcloud_item.field.shape[1],
        *random_pointcloud_item.field.shape[3:],
    )
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
    expected_field_shape = (
        random_pointcloud_item.field.shape[0],
        random_pointcloud_item.field.shape[1],
        *random_pointcloud_item.field.shape[3:],
    )
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
        _ = Rotate(rot_angle="invalid")  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_rotate_transform_invalid_rot_axis():
    with pytest.raises(ValueError):
        _ = Rotate(rot_axis="invalid")  # type: ignore[arg-type]  # Intentionally testing invalid input


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
    with patch("numpy.random.randint", return_value=2) as mock_randint:
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

    with patch("numpy.random.randint", return_value=0):
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

    with patch("numpy.random.randint", return_value=1):
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

    with patch("numpy.random.randint", return_value=2):
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

    with patch("numpy.random.randint", return_value=3):
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
        _ = Mirror(mirror_axis="invalid")  # type: ignore[arg-type]  # Intentionally testing invalid input


def test_mirror_transform_invalid_mirror_prob_type():
    with pytest.raises(ValueError):
        _ = Mirror(mirror_prob="invalid")  # type: ignore[arg-type]  # Intentionally testing invalid input


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


# Additional edge case tests for 100% coverage


def test_check_transforms_with_default_transform():
    """Test check_transforms passes for DefaultTransform without raising error."""
    aug = DefaultTransform()
    check_transforms(aug)


def test_check_transforms_with_single_phase_shift():
    """Test check_transforms passes for a single PhaseShift transform."""
    aug = PhaseShift(num_coils=8)
    check_transforms(aug)


def test_check_transforms_with_compose_no_phase_shift():
    """Test check_transforms raises error when Compose has no PhaseShift transform."""
    aug = Compose([FirstAugmentation(), SecondAugmentation()])
    with pytest.raises(ValueError, match="Exactly one of the composed transforms should be a PhaseShift transform"):
        check_transforms(aug)


def test_check_transforms_with_compose_multiple_phase_shifts():
    """Test check_transforms raises error when Compose has multiple PhaseShift transforms."""
    aug = Compose([PhaseShift(num_coils=8), PhaseShift(num_coils=8)])
    with pytest.raises(ValueError, match="Exactly one of the composed transforms should be a PhaseShift transform"):
        check_transforms(aug)


def test_check_transforms_with_invalid_transform_type():
    """Test check_transforms raises error for invalid transform types."""
    with pytest.raises(ValueError, match="Transforms not valid"):
        check_transforms(Crop(crop_size=(10, 10, 10)))


def test_check_transforms_with_not_base_transform():
    """Test check_transforms raises error when transform is not BaseTransform instance."""
    with pytest.raises(ValueError, match="Transforms should be an instance of BaseTransform"):
        check_transforms("not a transform")


def test_base_transform_call_not_implemented():
    """Test that BaseTransform raises NotImplementedError if __call__ is not properly overridden."""

    class MinimalTransform(BaseTransform):
        def __call__(self, simulation: DataItem):
            # Directly raise NotImplementedError without calling super()
            # to avoid mypy safe-super error on abstract method
            raise NotImplementedError

    transform = MinimalTransform()
    with pytest.raises(NotImplementedError):
        transform(None)  # type: ignore[arg-type]


def test_base_transform_repr():
    """Test BaseTransform.__repr__ returns correct string representation."""
    transform = FirstAugmentation()
    repr_str = repr(transform)
    assert repr_str == "FirstAugmentation{}"


def test_base_transform_repr_with_kwargs():
    """Test BaseTransform.__repr__ with kwargs."""
    transform = PhaseShift(num_coils=8, sampling_method="uniform")
    repr_str = repr(transform)
    assert "PhaseShift" in repr_str


def test_compose_repr():
    """Test Compose.__repr__ returns correct string representation."""
    aug = Compose([FirstAugmentation(), SecondAugmentation()])
    repr_str = repr(aug)
    assert "Compose" in repr_str
    assert "FirstAugmentation" in repr_str
    assert "SecondAugmentation" in repr_str


def test_default_transform_check_if_valid():
    """Test DefaultTransform.check_if_valid returns True."""
    transform = DefaultTransform()
    assert transform.check_if_valid() is True


def test_crop_unknown_position_edge_case(zero_grid_item):
    """Test Crop with manually set invalid crop_position raises error."""
    crop = Crop(crop_size=(10, 10, 10), crop_position="center")
    crop.crop_position = "invalid_position"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unknown crop position"):
        crop(zero_grid_item)


def test_phase_shift_sample_mask_binomial_all_zeros_retry(random_grid_item):
    """Test that _sample_mask_binomial retries when all zeros are sampled."""
    transform = PhaseShift(num_coils=8, sampling_method="binomial")

    with patch("numpy.random.choice") as mock_choice:
        mock_choice.side_effect = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),  # All zeros - triggers retry
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),  # Valid mask
        ]

        mask = transform._sample_mask_binomial()

        assert mock_choice.call_count == 2
        assert np.array_equal(mask, np.array([1, 0, 1, 0, 1, 0, 1, 0]))


def test_phase_shift_unknown_sampling_method_edge_case(random_grid_item):
    """Test PhaseShift with manually set invalid sampling_method raises error."""
    transform = PhaseShift(num_coils=8, sampling_method="uniform")
    transform.sampling_method = "invalid_method"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unknown sampling method"):
        transform._sample_phase_and_mask(dtype=random_grid_item.dtype)


def test_coil_enumerator_phase_shift_init():
    """Test CoilEnumeratorPhaseShift initialization."""
    transform = CoilEnumeratorPhaseShift(num_coils=8)

    assert transform.num_coils == 8
    assert transform.coil_on_index == 0


def test_coil_enumerator_phase_shift_sample_phase_zero():
    """Test CoilEnumeratorPhaseShift._sample_phase_zero returns zeros."""
    transform = CoilEnumeratorPhaseShift(num_coils=8)

    phase = transform._sample_phase_zero(dtype="float32")

    assert phase.shape == (8,)
    assert phase.dtype == np.float32
    assert np.all(phase == 0)


def test_coil_enumerator_phase_shift_sample_mask_single():
    """Test CoilEnumeratorPhaseShift._sample_mask_single cycles through coils."""
    transform = CoilEnumeratorPhaseShift(num_coils=8)

    mask1 = transform._sample_mask_single()
    expected1 = np.array([True, False, False, False, False, False, False, False])
    assert np.array_equal(mask1, expected1)
    assert transform.coil_on_index == 1

    mask2 = transform._sample_mask_single()
    expected2 = np.array([False, True, False, False, False, False, False, False])
    assert np.array_equal(mask2, expected2)
    assert transform.coil_on_index == 2

    transform.coil_on_index = 7
    mask_last = transform._sample_mask_single()
    expected_last = np.array([False, False, False, False, False, False, False, True])
    assert np.array_equal(mask_last, expected_last)
    assert transform.coil_on_index == 0


def test_coil_enumerator_phase_shift_sample_phase_and_mask():
    """Test CoilEnumeratorPhaseShift._sample_phase_and_mask returns correct values."""
    transform = CoilEnumeratorPhaseShift(num_coils=8)

    phase, mask = transform._sample_phase_and_mask(dtype="float32")

    assert phase.shape == (8,)
    assert mask.shape == (8,)
    assert phase.dtype == np.float32
    assert mask.dtype == np.bool_
    assert np.all(phase == 0)
    assert np.sum(mask) == 1


def test_coil_enumerator_phase_shift_full_cycle(random_grid_item):
    """Test CoilEnumeratorPhaseShift cycles through all coils correctly."""
    transform = CoilEnumeratorPhaseShift(num_coils=8)

    for i in range(8):
        result = transform(random_grid_item)

        assert np.all(result.phase == 0)
        assert np.sum(result.mask) == 1

        if i == 7:
            assert transform.coil_on_index == 0
        else:
            assert transform.coil_on_index == (i + 1)


def test_coil_enumerator_phase_shift_integration(random_grid_item):
    """Test CoilEnumeratorPhaseShift full integration with DataItem."""
    transform = CoilEnumeratorPhaseShift(num_coils=8)

    result = transform(random_grid_item)

    assert isinstance(result, DataItem)
    assert result.phase is not None and result.phase.shape == (8,)
    assert result.mask is not None and result.mask.shape == (8,)
    assert result.phase is not None and np.all(result.phase == 0)
    assert result.mask is not None and np.sum(result.mask) == 1
    assert result.field is not None and result.field.shape[2] == 3
    assert result.coils is not None and result.coils.shape[0] == 2
