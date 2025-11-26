import pytest
import torch
import einops

from magnet_pinn.losses.utils import DiffFilterFactory, LossReducer, ObjectMaskCropping
from magnet_pinn.losses.physics import DivergenceLoss, FaradaysLawLoss



def test_single_derivatives(tensor_values_changing_along_dim, diff_filter_factory, num_dims):
    values, dim = tensor_values_changing_along_dim
    values = values.unsqueeze(0).unsqueeze(0)
    for i in range(num_dims):
        filter_tensor = diff_filter_factory.derivative_from_expression(diff_filter_factory.dim_names[i])
        filter_tensor = einops.rearrange(filter_tensor, '... -> () () ...')

        derivative = torch.nn.functional.conv3d(values, filter_tensor)

        if dim != i:
            assert torch.allclose(
                derivative, torch.zeros_like(derivative), atol=1e-6
            ), f"Derivative is not zero along dimension {i}, but should be (field varies along {dim})"
        else:
            assert not torch.allclose(
                derivative, torch.zeros_like(derivative), atol=1e-6
            ), f"Derivative is zero along dimension {i}, but field varies along {dim}"


def test_equivalent_padding_twice(diff_filter_factory):
    tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    padded_tensor = diff_filter_factory._pad_to_square(tensor)
    padded_tensor2 = diff_filter_factory._pad_to_square(padded_tensor)
    assert torch.equal(padded_tensor, padded_tensor2), f"Expected {padded_tensor}, but got {padded_tensor2}"


def test_divergence_equivalence_single_derivatives(
    diff_filter_factory, tensor_random_ndim_field, num_dims
):
    """
    Test that divergence computed via divergence filter equals
    the sum of individual partial derivatives.
    """
    divergence_filter = diff_filter_factory.divergence()
    field = tensor_random_ndim_field

    # Compute divergence using the combined filter
    divergence = torch.nn.functional.conv3d(field, divergence_filter)

    # Compute divergence manually as sum of partial derivatives
    dim_names = diff_filter_factory.dim_names
    manual_divergence = torch.zeros_like(divergence)

    for i in range(num_dims):
        # Get derivative filter for dimension i
        deriv_filter = diff_filter_factory.derivative_from_expression(dim_names[i])
        deriv_filter = diff_filter_factory._pad_to_square(deriv_filter)
        deriv_filter = einops.rearrange(deriv_filter, '... -> () () ...')

        # Apply to i-th component of the field
        component_derivative = torch.nn.functional.conv3d(
            field[:, i:i+1, :, :, :], deriv_filter
        )

        manual_divergence = manual_divergence + component_derivative

    result_msg = (
        f"Divergence filter result differs from sum of partial derivatives.\n"
        f"Max difference: {torch.max(torch.abs(divergence - manual_divergence))}"
    )
    assert torch.allclose(divergence, manual_divergence, atol=1e-5), result_msg


def test_curl_on_constant_field(diff_filter_factory, num_values):
    constant_field = torch.ones([1, 3, num_values, num_values, num_values])
    curl_filter = diff_filter_factory.curl()
    curl_result = torch.nn.functional.conv3d(constant_field, curl_filter, padding=1)
    interior = curl_result[:, :, 2:-2, 2:-2, 2:-2]
    assert torch.allclose(
        interior, torch.zeros_like(interior), atol=1e-6
    )


def test_curl_components_on_linear_fields(diff_filter_factory, num_values):
    curl_filter = diff_filter_factory.curl()

    y_coords = torch.linspace(0, 1, num_values)
    field1 = torch.zeros([1, 3, num_values, num_values, num_values])
    field1[0, 2, :, :, :] = y_coords.view(1, -1, 1)
    curl1 = torch.nn.functional.conv3d(field1, curl_filter, padding=1)

    interior_slice = slice(2, -2)
    dy = (y_coords[-1] - y_coords[0]) / (num_values - 1)
    expected_curl_x = dy

    assert torch.allclose(
        curl1[0, 0, interior_slice, interior_slice, interior_slice],
        torch.full_like(curl1[0, 0, interior_slice, interior_slice, interior_slice], expected_curl_x),
        rtol=0.01
    )
    assert torch.allclose(
        curl1[0, 1, interior_slice, interior_slice, interior_slice],
        torch.zeros_like(curl1[0, 1, interior_slice, interior_slice, interior_slice]),
        atol=1e-6
    )
    assert torch.allclose(
        curl1[0, 2, interior_slice, interior_slice, interior_slice],
        torch.zeros_like(curl1[0, 2, interior_slice, interior_slice, interior_slice]),
        atol=1e-6
    )

    z_coords = torch.linspace(0, 1, num_values)
    field2 = torch.zeros([1, 3, num_values, num_values, num_values])
    field2[0, 0, :, :, :] = z_coords.view(1, 1, -1)
    curl2 = torch.nn.functional.conv3d(field2, curl_filter, padding=1)

    dz = (z_coords[-1] - z_coords[0]) / (num_values - 1)
    expected_curl_y = dz

    assert torch.allclose(
        curl2[0, 0, interior_slice, interior_slice, interior_slice],
        torch.zeros_like(curl2[0, 0, interior_slice, interior_slice, interior_slice]),
        atol=1e-6
    )
    assert torch.allclose(
        curl2[0, 1, interior_slice, interior_slice, interior_slice],
        torch.full_like(curl2[0, 1, interior_slice, interior_slice, interior_slice], expected_curl_y),
        rtol=0.01
    )
    assert torch.allclose(
        curl2[0, 2, interior_slice, interior_slice, interior_slice],
        torch.zeros_like(curl2[0, 2, interior_slice, interior_slice, interior_slice]),
        atol=1e-6
    )

    x_coords = torch.linspace(0, 1, num_values)
    field3 = torch.zeros([1, 3, num_values, num_values, num_values])
    field3[0, 1, :, :, :] = x_coords.view(-1, 1, 1)
    curl3 = torch.nn.functional.conv3d(field3, curl_filter, padding=1)

    dx = (x_coords[-1] - x_coords[0]) / (num_values - 1)
    expected_curl_z = dx

    assert torch.allclose(
        curl3[0, 0, interior_slice, interior_slice, interior_slice],
        torch.zeros_like(curl3[0, 0, interior_slice, interior_slice, interior_slice]),
        atol=1e-6
    )
    assert torch.allclose(
        curl3[0, 1, interior_slice, interior_slice, interior_slice],
        torch.zeros_like(curl3[0, 1, interior_slice, interior_slice, interior_slice]),
        atol=1e-6
    )
    assert torch.allclose(
        curl3[0, 2, interior_slice, interior_slice, interior_slice],
        torch.full_like(curl3[0, 2, interior_slice, interior_slice, interior_slice], expected_curl_z),
        rtol=0.01
    )


def test_divergence_on_zero_divergence_field(diff_filter_factory):
    """
    Test divergence on a field with known zero divergence.
    For a solenoidal field like V = (y, -x, 0), div(V) should be 0.
    """
    num_values = 50
    divergence_filter = diff_filter_factory.divergence()

    x = torch.linspace(-1, 1, num_values)
    y = torch.linspace(-1, 1, num_values)
    z = torch.linspace(-1, 1, num_values)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    field = torch.zeros([1, 3, num_values, num_values, num_values])
    field[0, 0] = Y
    field[0, 1] = -X
    field[0, 2] = 0

    div_result = torch.nn.functional.conv3d(field, divergence_filter)

    assert torch.allclose(
        div_result, torch.zeros_like(div_result), atol=1e-4
    )


def test_masked_loss_reducer_shape_mismatch():
    reducer = LossReducer()
    loss = torch.randn(4, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="Loss shape and mask shape are different: .* != .*"):
        reducer(loss, mask)


def test_masked_loss_reducer_with_none_mask():
    reducer = LossReducer()
    loss = torch.randn(4, 8)

    result = reducer(loss, None)

    assert result.shape == torch.Size([])
    assert torch.isfinite(result)


def test_object_mask_padding():
    padding_fn = ObjectMaskCropping(padding=1)
    mask = torch.ones([1, 1, 10, 10, 10])
    mask[:, :, 2:8, 2:8, 2:8] = 0

    result = padding_fn(mask)

    assert result.shape[0] == 1
    assert result.shape[1] == 1
    assert result.dtype == torch.bool


def test_diff_filter_factory_invalid_dim():
    factory = DiffFilterFactory(num_dims=3)

    with pytest.raises(ValueError, match="dim .* must be less than num_dims"):
        factory._generate_einops_expansion_expression(5)


def test_diff_filter_factory_mismatched_dim_names():
    with pytest.raises(ValueError, match="dim_names .* does not match num_dims"):
        DiffFilterFactory(num_dims=3, dim_names='xy')



def test_mask_padding_default_padding():
    mask = torch.ones([1, 1, 10, 10, 10])
    mask[:, :, 3:7, 3:7, 3:7] = 0
    
    mask_padding = ObjectMaskCropping()

    result = mask_padding(mask)

    assert result.dtype == torch.bool
    assert result.shape == mask.shape


def test_diff_filter_factory_with_non_default_dx():
    """Test that DiffFilterFactory correctly scales derivatives with non-default dx."""
    dx = 0.5
    factory = DiffFilterFactory(dx=dx, num_dims=3)

    num_values = 21
    x_coords = torch.linspace(0, 10, num_values)
    field = torch.zeros([1, 3, num_values, num_values, num_values])
    field[0, 0, :, :, :] = x_coords.view(-1, 1, 1)

    div_filter = factory.divergence()
    div_result = torch.nn.functional.conv3d(field, div_filter, padding=1)

    grid_dx = (x_coords[-1] - x_coords[0]) / (num_values - 1)
    expected_divergence = grid_dx / dx

    interior_slice = slice(2, -2)
    interior = div_result[0, 0, interior_slice, interior_slice, interior_slice]

    assert torch.allclose(interior, torch.full_like(interior, float(expected_divergence)), rtol=0.01)


def test_diff_filter_factory_invalid_negative_dx():
    """Test that DiffFilterFactory raises error for negative dx."""
    with pytest.raises(ValueError):
        DiffFilterFactory(dx=-1.0)


def test_diff_filter_factory_invalid_zero_dx():
    """Test that DiffFilterFactory raises error for zero dx."""
    with pytest.raises(ValueError):
        DiffFilterFactory(dx=0.0)


def test_diff_filter_factory_invalid_accuracy():
    """Test that DiffFilterFactory handles invalid accuracy values appropriately."""
    with pytest.raises(ValueError):
        DiffFilterFactory(accuracy=0)

    with pytest.raises(ValueError):
        DiffFilterFactory(accuracy=-2)


def test_diff_filter_factory_padding_calculation():
    """Test that padding calculation is correct for different accuracy levels."""
    factory_acc2 = DiffFilterFactory(accuracy=2, num_dims=3)
    div_filter_acc2 = factory_acc2.divergence()
    assert div_filter_acc2.shape[-1] == 3
    expected_padding_acc2 = 1
    assert expected_padding_acc2 == factory_acc2.accuracy // 2

    factory_acc4 = DiffFilterFactory(accuracy=4, num_dims=3)
    div_filter_acc4 = factory_acc4.divergence()
    assert div_filter_acc4.shape[-1] == 5
    expected_padding_acc4 = 2
    assert expected_padding_acc4 == factory_acc4.accuracy // 2


def test_divergence_loss_uses_correct_padding():
    """Test that DivergenceLoss uses dynamically calculated padding."""
    loss_fn = DivergenceLoss()

    assert loss_fn.diff_filter_factory.accuracy == 2

    expected_padding = loss_fn.diff_filter_factory.accuracy // 2
    assert expected_padding == 1

    spatial_size = 16
    batch_size = 1
    pred = torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size)
    target = torch.zeros_like(pred)

    loss = loss_fn(pred, target)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_faradays_loss_uses_correct_padding(random_fields):
    """Test that FaradaysLoss uses dynamically calculated padding."""
    loss_fn = FaradaysLawLoss()

    assert loss_fn.diff_filter_factory.accuracy == 2

    expected_padding = loss_fn.diff_filter_factory.accuracy // 2
    assert expected_padding == 1

    loss = loss_fn(random_fields)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
