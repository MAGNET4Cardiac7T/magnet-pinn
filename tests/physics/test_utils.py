import pytest
import torch
from magnet_pinn.losses.utils import DiffFilterFactory, MaskedLossReducer, ObjectMaskPadding
import einops


@pytest.fixture
def diff_filter_factory(num_dims) -> DiffFilterFactory:
    return DiffFilterFactory(num_dims=num_dims)


@pytest.fixture
def num_dims() -> int:
    return 3


@pytest.fixture
def num_values() -> int:
    return 100


@pytest.fixture
def alphabet() -> str:
    return 'bcdefghijklmnopqrstuvwxyz'


@pytest.fixture(params=[0, 1, 2])
def tensor_values_changing_along_dim(request, num_dims, num_values, alphabet):
    values = torch.linspace(0, 1, num_values)
    dim = request.param
    dims_before = ' '.join([alphabet[i] for i in range(dim)])
    dims_after = ' '.join([alphabet[i] for i in range(dim+1, num_dims)])
    repeat_pattern = 'a -> ' + dims_before + ' a ' + dims_after
    repeats_dict = {alphabet[d]: len(values) for d in range(num_dims) if d != dim}
    return einops.repeat(values, repeat_pattern, **repeats_dict), dim


@pytest.fixture(params=[42, 43, 44])
def tensor_random_ndim_field(request, num_dims, num_values):
    torch.manual_seed(request.param)
    return torch.rand([1, num_dims] + num_dims*[num_values])


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
        rtol=0.02
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
        rtol=0.02
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
        rtol=0.02
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
    reducer = MaskedLossReducer()
    loss = torch.randn(4, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape .* does not match loss shape"):
        reducer(loss, mask)


def test_masked_loss_reducer_with_none_mask():
    reducer = MaskedLossReducer()
    loss = torch.randn(4, 8)

    result = reducer(loss, None)

    assert result.shape == torch.Size([])
    assert torch.isfinite(result)


def test_object_mask_padding():
    padding_fn = ObjectMaskPadding(padding=1)
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
