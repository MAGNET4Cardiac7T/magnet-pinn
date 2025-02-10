import numpy as np
from copy import copy
from einops import rearrange

from magnet_pinn.data._base import BaseTransform
from magnet_pinn.data.dataitem import DataItem


class FirstAugmentation(BaseTransform):
    def __call__(self, simulation: DataItem) -> DataItem:
        result = copy(simulation)
        result.simulation += "1"
        return result


class SecondAugmentation(BaseTransform):
    def __call__(self, simulation: DataItem) -> DataItem:
        result = copy(simulation)
        result.simulation += "2"
        return result


class ThirdAugmentation(BaseTransform):
    def __call__(self, simulation: DataItem) -> DataItem:
        result = copy(simulation)
        result.simulation += "3"
        return result
    

def check_items_datatypes(result, random_item):
    assert type(result.simulation) == type(random_item.simulation)
    assert result.input.dtype == random_item.input.dtype
    assert result.field.dtype == random_item.field.dtype
    assert result.subject.dtype == random_item.subject.dtype
    assert result.positions.dtype == random_item.positions.dtype
    assert result.phase.dtype == random_item.phase.dtype
    assert result.mask.dtype == random_item.mask.dtype
    assert result.coils.dtype == random_item.coils.dtype
    assert result.dtype == random_item.dtype
    assert type(result.dtype) == type(random_item.dtype)
    assert result.truncation_coefficients.dtype == random_item.truncation_coefficients.dtype


def check_cropped_shapes(result):
    assert result.input.shape == (3, 10, 10, 10)
    assert result.field.shape == (2, 2, 3, 10, 10, 10, 8)
    assert result.subject.shape[:-1] == (10, 10, 10)
    assert result.positions.shape == (0,)
    assert result.phase.shape == (8,)
    assert result.mask.shape == (8,)
    assert result.coils.shape == (10, 10, 10, 8)
    assert result.truncation_coefficients.shape == (3,)


def check_items_shapes_suppsed_to_be_equal(result, input_item):
    assert result.input.shape == input_item.input.shape
    assert result.field.shape == input_item.field.shape
    assert result.subject.shape == input_item.subject.shape
    assert result.positions.shape == input_item.positions.shape
    assert result.phase.shape == input_item.phase.shape
    assert result.mask.shape == input_item.mask.shape
    assert result.coils.shape == input_item.coils.shape
    assert result.truncation_coefficients.shape == input_item.truncation_coefficients.shape


def check_elements_not_changed_by_crop(result, input_item):
    assert result.simulation == input_item.simulation
    assert np.equal(result.positions, input_item.positions).all()
    assert np.equal(result.phase, input_item.phase).all()
    assert np.equal(result.mask, input_item.mask).all()
    assert result.dtype == input_item.dtype
    assert np.equal(result.truncation_coefficients, input_item.truncation_coefficients).all()


def check_constant_shapes_not_changed_except_for_field_coils(result, item): 
    assert len(result.simulation) == len(item.simulation)
    assert result.input.shape == item.input.shape
    assert result.subject.shape == item.subject.shape
    assert result.positions.shape == item.positions.shape
    assert result.phase.shape == item.phase.shape
    assert result.mask.shape == item.mask.shape
    assert len(result.dtype) == len(item.dtype)
    assert result.truncation_coefficients.shape == item.truncation_coefficients.shape


def check_constant_values_not_changed_by_phase_shift(result, item):
    assert result.simulation == item.simulation
    assert np.equal(result.input, item.input).all()
    assert np.equal(result.subject, item.subject).all()
    assert np.equal(result.positions, item.positions).all()
    assert result.dtype == item.dtype
    assert np.equal(result.truncation_coefficients, item.truncation_coefficients).all()


def check_default_transform_resulting_shapes(result, item):
    assert result.input.shape == item.input.shape
    assert result.subject.shape == item.subject.shape
    assert result.positions.shape == item.positions.shape
    assert result.phase.shape == item.phase.shape
    assert result.mask.shape == item.mask.shape

    assert result.field.shape == item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(item.coils.shape[:-1]))


def check_default_transform_resulting_values(result, item):
    assert result.simulation == item.simulation
    assert np.equal(result.input, item.input).all()
    assert np.equal(result.subject, item.subject).all()
    assert np.equal(result.positions, item.positions).all()
    assert result.dtype == item.dtype
    assert np.equal(result.truncation_coefficients, item.truncation_coefficients).all()

    assert np.equal(result.field, np.sum(item.field, axis=-1)).all()

    coils_num = item.coils.shape[-1]
    assert np.equal(result.phase, np.zeros(coils_num, dtype=item.phase.dtype)).all()
    assert np.equal(result.mask, np.ones(coils_num, dtype=item.mask.dtype)).all()

    expected_coils = np.stack([
        np.sum(item.coils, axis=-1),
        np.zeros(item.coils.shape[:-1], dtype=item.coils.dtype)
    ], axis=0)
    assert np.equal(result.coils, expected_coils).all()


def check_complex_number_calculations_in_phase_shift(result, item):
    coefs_re = np.cos(result.phase) * result.mask
    coefs_im = np.sin(result.phase) * result.mask

    field_re = item.field[:, 0]
    field_im = item.field[:, 1]

    field_shifted_re = field_re @ coefs_re - field_im @ coefs_im
    field_shifted_im = field_re @ coefs_im + field_im @ coefs_re

    expected_field_result = np.stack([field_shifted_re, field_shifted_im], axis=1)
    assert np.equal(result.field, expected_field_result).all()

    coils_re = item.coils @ coefs_re
    coils_im = item.coils @ coefs_im

    expected_coils_result = np.stack([coils_re, coils_im], axis=0)
    assert np.equal(result.coils, expected_coils_result).all()


def check_complex_number_calculations_in_pointscloud_phase_shift(result, item):
    """
    This function assumes the preprocessing did not standartize the axis position and `fieldxyz` and `positions` are having different order
    """
    coefs_re = np.cos(result.phase) * result.mask
    coefs_im = np.sin(result.phase) * result.mask

    field_re = item.field[:, 0]
    field_im = item.field[:, 1]

    field_shifted_re = field_re @ coefs_re - field_im @ coefs_im
    field_shifted_im = field_re @ coefs_im + field_im @ coefs_re

    expected_field_result = np.stack([field_shifted_re, field_shifted_im], axis=1)
    expected_field_result = np.ascontiguousarray(
        rearrange(expected_field_result, "he reim position fieldxyz -> he reim fieldxyz position")
    ).astype(np.float32)
    assert np.equal(result.field, expected_field_result).all()

    coils_re = item.coils @ coefs_re
    coils_im = item.coils @ coefs_im

    expected_coils_result = np.stack([coils_re, coils_im], axis=0)
    assert np.equal(result.coils, expected_coils_result).all()


def check_pointcloud_feature_rearrange_shapes_field_coils(result, item):
    assert result.field.shape == tuple(item.field.shape[-1::-1])
    assert result.coils.shape == tuple(item.coils.shape[-1::-1])


def check_constant_values_not_changed_except_for_field_coils(result, item):
    assert result.simulation == item.simulation
    assert np.equal(result.input, item.input).all()
    assert np.equal(result.subject, item.subject).all()
    assert np.equal(result.positions, item.positions).all()
    assert np.equal(result.phase, item.phase).all()
    assert np.equal(result.mask, item.mask).all()
    assert result.dtype == item.dtype
    assert np.equal(result.truncation_coefficients, item.truncation_coefficients).all()


def check_pointcloud_feature_rearrange_values_field_coils(result, item):
    expected_field_array = np.ascontiguousarray(rearrange(
        item.field, "he reim fieldxyz positions -> positions fieldxyz reim he"
    )).astype(np.float32)
    assert np.equal(result.field, expected_field_array).all()

    expected_coils_array = np.ascontiguousarray(rearrange(
        item.coils, "reim positions -> positions reim"
    ))
    assert np.equal(result.coils, expected_coils_array).all()
