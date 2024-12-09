import pytest
import torch
from magnet_pinn.losses.utils import DiffFilterFactory
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

@pytest.fixture(params=[0,1,2])
def tensor_values_changing_along_dim(request, num_dims, num_values, alphabet) -> torch.Tensor:
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
    return torch.rand([num_dims] + num_dims*[num_values])


def test_single_derivatives(tensor_values_changing_along_dim, diff_filter_factory, num_dims):
    values, dim = tensor_values_changing_along_dim
    values = values.unsqueeze(0)
    for i in range(num_dims):
        filter_tensor = diff_filter_factory.derivative_from_expression(diff_filter_factory.dim_names[i])
        filter_tensor = einops.rearrange(filter_tensor, '... -> () () ...')

        
        derivative = torch.nn.functional.conv3d(values, filter_tensor)


        
        if dim != i:
            # derivative should be zero along all dimension that is not changing
            assert torch.allclose(derivative, torch.zeros_like(derivative), atol=1e-6), f"Derivative is not zero along dimension {dim}"
        else:
            # derivative should not be zero along the dimension that is changing
            assert not torch.allclose(derivative, torch.zeros_like(derivative), atol=1e-6), f"Derivative is zero along dimension {dim}"

def test_equivalent_padding_twice(diff_filter_factory):
    tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    padded_tensor = diff_filter_factory._pad_to_square(tensor)
    padded_tensor2 = diff_filter_factory._pad_to_square(padded_tensor)
    assert torch.equal(padded_tensor, padded_tensor2), f"Expected {padded_tensor}, but got {padded_tensor2}"

def test_divergence_equivalence_single_derivatives(diff_filter_factory, tensor_random_ndim_field, num_values, num_dims):
    # applying the divergence operator should be equivalent to applying the single derivative operator along each dimension and adding up
    divergence_filter = diff_filter_factory.divergence()
    field = tensor_random_ndim_field
    divergence = torch.nn.functional.conv3d(field, divergence_filter)
    dim_names = diff_filter_factory.dim_names

    single_derivative_filters = [diff_filter_factory.derivative_from_expression(dim_names[i]) for i in range(num_dims)]
    single_derivative_filters = [einops.rearrange(filter_tensor, '... -> () () ...') for filter_tensor in single_derivative_filters]

    single_derivatives = []
    for i in range(num_dims):
        derivative = torch.nn.functional.conv3d(field[i:i+1], single_derivative_filters[i])
        padding = [0]*2*num_dims
        padding[-(2*i+2):-(2*i)] = 1, 1
        derivative = torch.nn.functional.pad(derivative, padding, mode='constant', value=0)
        single_derivatives.append(derivative)

    divergence_from_single_derivatives = sum(single_derivatives)
    negative_padding = [-1]*2*num_dims
    divergence_from_single_derivatives = torch.nn.functional.pad(divergence_from_single_derivatives, negative_padding, mode='constant', value=0)

    assert torch.allclose(divergence, divergence_from_single_derivatives, atol=1e-6), f"Expected {divergence}, but got {divergence_from_single_derivatives}"
