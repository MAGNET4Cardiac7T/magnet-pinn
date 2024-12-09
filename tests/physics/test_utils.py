import pytest
import torch
from magnet_pinn.losses.utils import DiffFilterFactory
import einops


@pytest.fixture
def diff_filter_factory(num_dims):
    return DiffFilterFactory(num_dims=num_dims)

@pytest.fixture
def num_dims():
    return 3

@pytest.fixture
def num_values():
    return 100

@pytest.fixture
def alphabet():
    return 'bcdefghijklmnopqrstuvwxyz'

@pytest.fixture(params=[0,1,2])
def values_changing_along_dim(request, num_dims, num_values, alphabet):
    values = torch.linspace(0, 1, num_values)
    dim = request.param
    dims_before = ' '.join([alphabet[i] for i in range(dim)])
    dims_after = ' '.join([alphabet[i] for i in range(dim+1, num_dims)])
    repeat_pattern = 'a -> ' + dims_before + ' a ' + dims_after
    repeats_dict = {alphabet[d]: len(values) for d in range(num_dims) if d != dim}
    return einops.repeat(values, repeat_pattern, **repeats_dict), dim

def test_single_derivatives(values_changing_along_dim, diff_filter_factory, num_dims):
    values, dim = values_changing_along_dim
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
