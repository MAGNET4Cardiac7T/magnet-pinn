from copy import deepcopy

import numpy as np
import pytest

from magnet_pinn.data.dataitem import AugmentedDataItem, DataItem
from magnet_pinn.data.transforms import (
    AsDict,
    B1PlusTransform,
    Compose,
    PhaseShift,
    PointwiseSARTransform,
)


# --- B1PlusTransform tests ---


def test_b1plus_transform_invalid_input():
    with pytest.raises(ValueError):
        B1PlusTransform()(None)


def test_b1plus_transform_returns_augmented_data_item(phase_shifted_grid_item):
    result = B1PlusTransform()(phase_shifted_grid_item)
    assert isinstance(result, AugmentedDataItem)


def test_b1plus_transform_output_shape(phase_shifted_grid_item):
    result = B1PlusTransform()(phase_shifted_grid_item)
    # B1+ should have shape (2, spatial_axis...) — real and imaginary parts
    spatial_shape = phase_shifted_grid_item.field.shape[3:]
    assert result.b1plus.shape == (2, *spatial_shape)


def test_b1plus_transform_preserves_original_fields(phase_shifted_grid_item):
    item = deepcopy(phase_shifted_grid_item)
    result = B1PlusTransform()(item)

    assert np.array_equal(result.input, item.input)
    assert np.array_equal(result.field, item.field)
    assert np.array_equal(result.subject, item.subject)
    assert result.simulation == item.simulation
    assert np.array_equal(result.positions, item.positions)
    assert np.array_equal(result.phase, item.phase)
    assert np.array_equal(result.mask, item.mask)
    assert np.array_equal(result.coils, item.coils)


def test_b1plus_transform_calculation(phase_shifted_grid_item):
    """Verify B1+ = 0.5 * (Bx + j*By) from H-field."""
    result = B1PlusTransform()(phase_shifted_grid_item)

    b_field = phase_shifted_grid_item.field[1]  # H-field
    b_field_complex = b_field[0] + 1j * b_field[1]  # re + j*im
    b1_plus = 0.5 * (b_field_complex[0] + 1j * b_field_complex[1])  # Bx + j*By

    np.testing.assert_allclose(result.b1plus[0], b1_plus.real, rtol=1e-5)
    np.testing.assert_allclose(result.b1plus[1], b1_plus.imag, rtol=1e-5)


def test_b1plus_transform_not_inplace(phase_shifted_grid_item):
    result = B1PlusTransform()(phase_shifted_grid_item)
    assert result is not phase_shifted_grid_item


def test_b1plus_transform_preserves_existing_sar(augmented_grid_item):
    """When input is already an AugmentedDataItem with SAR, B1Plus should preserve it."""
    result = B1PlusTransform()(augmented_grid_item)
    assert isinstance(result, AugmentedDataItem)
    np.testing.assert_array_equal(result.sar, augmented_grid_item.sar)


# --- PointwiseSARTransform tests ---


def test_pointwise_sar_transform_invalid_input():
    with pytest.raises(ValueError):
        PointwiseSARTransform()(None)


def test_pointwise_sar_transform_returns_augmented_data_item(phase_shifted_grid_item):
    result = PointwiseSARTransform()(phase_shifted_grid_item)
    assert isinstance(result, AugmentedDataItem)


def test_pointwise_sar_transform_output_shape(phase_shifted_grid_item):
    result = PointwiseSARTransform()(phase_shifted_grid_item)
    spatial_shape = phase_shifted_grid_item.field.shape[3:]
    assert result.sar.shape == (1, *spatial_shape)


def test_pointwise_sar_transform_preserves_original_fields(phase_shifted_grid_item):
    item = deepcopy(phase_shifted_grid_item)
    result = PointwiseSARTransform()(item)

    assert np.array_equal(result.input, item.input)
    assert np.array_equal(result.field, item.field)
    assert np.array_equal(result.subject, item.subject)
    assert result.simulation == item.simulation


def test_pointwise_sar_transform_calculation(phase_shifted_grid_item):
    """Verify SAR = sigma * |E|^2 / rho."""
    result = PointwiseSARTransform()(phase_shifted_grid_item)

    e_field = phase_shifted_grid_item.field[0]
    abs_efield_sq = np.sum(e_field**2, axis=(0, 1))
    conductivity = phase_shifted_grid_item.input[0]
    density = phase_shifted_grid_item.input[2]
    subject = phase_shifted_grid_item.subject
    expected_sar = subject * conductivity * abs_efield_sq / density

    np.testing.assert_allclose(result.sar[0], expected_sar, rtol=1e-5)


def test_pointwise_sar_transform_not_inplace(phase_shifted_grid_item):
    result = PointwiseSARTransform()(phase_shifted_grid_item)
    assert result is not phase_shifted_grid_item


def test_pointwise_sar_transform_preserves_existing_b1plus(augmented_grid_item):
    """When input is already an AugmentedDataItem with B1+, SAR should preserve it."""
    result = PointwiseSARTransform()(augmented_grid_item)
    assert isinstance(result, AugmentedDataItem)
    np.testing.assert_array_equal(result.b1plus, augmented_grid_item.b1plus)


# --- AsDict transform tests ---


def test_asdict_transform_invalid_input():
    with pytest.raises(ValueError):
        AsDict()(None)


def test_asdict_transform_returns_dict(phase_shifted_grid_item):
    result = AsDict()(phase_shifted_grid_item)
    assert isinstance(result, dict)


def test_asdict_transform_contains_all_dataitem_keys(phase_shifted_grid_item):
    result = AsDict()(phase_shifted_grid_item)
    for key in ["simulation", "input", "field", "subject", "positions", "phase", "mask", "coils", "dtype", "truncation_coefficients"]:
        assert key in result


def test_asdict_transform_contains_augmented_keys(augmented_grid_item):
    result = AsDict()(augmented_grid_item)
    assert "b1plus" in result
    assert "sar" in result


def test_asdict_transform_values_match(phase_shifted_grid_item):
    result = AsDict()(phase_shifted_grid_item)
    assert result["simulation"] == phase_shifted_grid_item.simulation
    np.testing.assert_array_equal(result["input"], phase_shifted_grid_item.input)
    np.testing.assert_array_equal(result["field"], phase_shifted_grid_item.field)


# --- Compose integration tests ---


def test_compose_b1plus_and_sar(phase_shifted_grid_item):
    """B1+ and SAR transforms can be composed together."""
    transform = Compose([
        PhaseShift(num_coils=8),
        B1PlusTransform(),
        PointwiseSARTransform(),
    ])
    # Need pre-PhaseShift data for this; use the fixture that already has post-PhaseShift shape.
    # Instead, just test the two derived transforms composed:
    item = deepcopy(phase_shifted_grid_item)
    b1 = B1PlusTransform()(item)
    result = PointwiseSARTransform()(b1)

    assert isinstance(result, AugmentedDataItem)
    assert result.b1plus.shape[0] == 2
    assert result.sar.shape[0] == 1


def test_compose_b1plus_sar_asdict(phase_shifted_grid_item):
    """Full pipeline: B1+ -> SAR -> AsDict produces dict with all keys."""
    item = deepcopy(phase_shifted_grid_item)
    b1 = B1PlusTransform()(item)
    sar = PointwiseSARTransform()(b1)
    result = AsDict()(sar)

    assert isinstance(result, dict)
    assert "b1plus" in result
    assert "sar" in result
    assert "input" in result
    assert "field" in result
