import pytest
import torch
import einops

from magnet_pinn.losses.utils import DiffFilterFactory

@pytest.fixture(scope="module")
def batch_size():
    return 1

@pytest.fixture(scope="module")
def spatial_size():
    return 16

@pytest.fixture(scope="function")
def random_fields(batch_size, spatial_size):
    return tuple(
        torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size, dtype=torch.float32)
        for _ in range(4)
    )

@pytest.fixture(scope="function")
def random_fields_float64(batch_size, spatial_size):
    return tuple(
        torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size, dtype=torch.float64)
        for _ in range(4)
    )

@pytest.fixture(scope="function")
def random_fields_with_gradient(batch_size, spatial_size):
    fields = tuple(
        torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size, dtype=torch.float32, requires_grad=True)
        for _ in range(4)
    )
    return fields

@pytest.fixture(scope="function")
def zero_fields(batch_size, spatial_size):
    return tuple(
        torch.zeros(batch_size, 3, spatial_size, spatial_size, spatial_size, dtype=torch.float32)
        for _ in range(4)
    )

@pytest.fixture(scope="function")
def violating_fields(batch_size, spatial_size):
    return (
        torch.ones(batch_size, 3, spatial_size, spatial_size, spatial_size),
        torch.ones(batch_size, 3, spatial_size, spatial_size, spatial_size) * 0.5,
        torch.ones(batch_size, 3, spatial_size, spatial_size, spatial_size) * 0.1,
        torch.ones(batch_size, 3, spatial_size, spatial_size, spatial_size) * 0.05,
    )

@pytest.fixture(scope="function")
def half_mask(batch_size, spatial_size):
    mask = torch.ones(batch_size, spatial_size, spatial_size, spatial_size, dtype=torch.bool)
    mask[:, :spatial_size//2] = False
    return mask




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
