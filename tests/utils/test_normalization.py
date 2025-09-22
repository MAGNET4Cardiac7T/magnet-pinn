import pytest
import torch
import numpy as np
from magnet_pinn.utils._normalization import (
    Identity, Power, Log, Tanh, Arcsinh,
    MinMaxNormalizer, StandardNormalizer
)

def test_random_batch(random_iterator, random_batch):
    iterator = random_iterator(seed=42, num_batches=20, batch_size=10, num_features=3)
    first_batch = next(iterator)["input"]
    assert torch.allclose(random_batch, first_batch), "random_batch does not match the first batch of random_iterator"
    assert isinstance(random_batch, torch.Tensor)
    assert random_batch.ndim == 2

@pytest.mark.parametrize("nonlinearity_class", [Identity, Power, Log, Tanh, Arcsinh])
def test_nonlinearity_forward_inverse(nonlinearity_class):
    if nonlinearity_class == Power:
        nonlinearity = nonlinearity_class(power=2.0)
    else:
        nonlinearity = nonlinearity_class()
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = nonlinearity.forward(x)
    x_reconstructed = nonlinearity.inverse(y)
    assert torch.allclose(x, x_reconstructed, atol=1e-6)

def test_minmax_normalizer_fit(random_iterator,random_batch):
    normalizer = MinMaxNormalizer()
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    normalized = normalizer.forward(random_batch, axis=1)
    denormalized = normalizer.inverse(normalized, axis=1)
    assert torch.allclose(random_batch, denormalized, atol=1e-6)

def test_standard_normalizer_fit(random_iterator, random_batch):
    normalizer = StandardNormalizer()
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    normalized = normalizer.forward(random_batch, axis=1)
    denormalized = normalizer.inverse(normalized, axis=1)
    assert torch.allclose(random_batch, denormalized, atol=1e-6)    

def test_minmax_normalizer_param_shape(random_iterator):
    normalizer = MinMaxNormalizer()
    num_features = 3
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=num_features)
    normalizer.fit_params(iterator, axis=1)

    params = normalizer.params
    assert "x_min" in params and "x_max" in params
    assert len(params["x_min"]) == num_features
    assert len(params["x_max"]) == num_features

def test_standard_normalizer_param_shape(random_iterator):
    normalizer = StandardNormalizer()
    num_features = 3
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=num_features)
    normalizer.fit_params(iterator, axis=1)

    params = normalizer.params
    assert "x_mean" in params and "x_mean_sq" in params
    assert len(params["x_mean"]) == num_features
    assert len(params["x_mean_sq"]) == num_features

def test_minmax_normallizer_correctness(random_iterator,random_batch):
    normalizer = MinMaxNormalizer()
    iterator = random_iterator(seed=42, num_batches=1, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    normalized = normalizer.forward(random_batch, axis=1)
    assert torch.allclose(normalized.min(dim=0).values, torch.tensor(0.0))
    assert torch.allclose(normalized.max(dim=0).values, torch.tensor(1.0))

    denormalized = normalizer.inverse(normalized, axis=1)
    assert torch.allclose(random_batch, denormalized, atol=1e-6)

def test_standard_normalizer_correctness(random_iterator, random_batch):
    normalizer = StandardNormalizer()
    iterator = random_iterator(seed=42, num_batches=1, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    normalized = normalizer.forward(random_batch, axis=1)
    assert torch.allclose(normalized.mean(dim=0), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized.std(dim=0), torch.tensor(1.0), atol=1e-6)

    denormalized = normalizer.inverse(normalized, axis=1)
    assert torch.allclose(random_batch, denormalized, atol=1e-6)

def test_save_and_load_minmax_normalizer(tmp_path, random_iterator):
    normalizer = MinMaxNormalizer()
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)
    
    original_params = normalizer.params.copy()  # Save original parameters
    save_path = tmp_path / "minmax_normalizer.json"
    normalizer.save_as_json(str(save_path))
    
    loaded_normalizer = MinMaxNormalizer.load_from_json(str(save_path))
    loaded_params = loaded_normalizer.params  # Load parameters from the saved file
    
    assert original_params == loaded_params, "Loaded parameters do not match the original parameters"

def test_save_and_load_standard_normalizer(tmp_path):
    normalizer = StandardNormalizer()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalizer.fit_params([{"input": data}])
    
    original_params = normalizer.params.copy()  # Save original parameters
    save_path = tmp_path / "standard_normalizer.json"
    normalizer.save_as_json(str(save_path))
    
    loaded_normalizer = StandardNormalizer.load_from_json(str(save_path))
    loaded_params = loaded_normalizer.params  # Load parameters from the saved file
    
    assert original_params == loaded_params, "Loaded parameters do not match the original parameters"
