import pytest
import torch
import numpy as np
from magnet_pinn.utils._normalization import (
    Identity, Power, Log, Tanh, Arcsinh,
    MinMaxNormalizer, StandardNormalizer, MetaNormalizer
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

def test_minmax_normalizer_correctness(random_iterator,random_batch):
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
    assert torch.allclose(normalized.std(dim=0, correction=0), torch.tensor(1.0), atol=1e-6)

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

def test_save_and_load_standard_normalizer(tmp_path, random_iterator):
    normalizer = StandardNormalizer()
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)
    
    original_params = normalizer.params.copy()  # Save original parameters
    save_path = tmp_path / "standard_normalizer.json"
    normalizer.save_as_json(str(save_path))
    
    loaded_normalizer = StandardNormalizer.load_from_json(str(save_path))
    loaded_params = loaded_normalizer.params  # Load parameters from the saved file
    
    assert original_params == loaded_params, "Loaded parameters do not match the original parameters"

def test_meta_normalizer_fit(random_iterator, random_batch):
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = MetaNormalizer([minmax_normalizer, standard_normalizer])

    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    meta_normalizer.fit_params(iterator, axis=1, keys="input")

    # Fit individual normalizers for comparison
    minmax_normalizer_individual = MinMaxNormalizer()
    standard_normalizer_individual = StandardNormalizer()
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    minmax_normalizer_individual.fit_params(iterator, axis=1)
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    standard_normalizer_individual.fit_params(iterator, axis=1)

    # Compare parameters
    assert minmax_normalizer.params == minmax_normalizer_individual.params, "MinMaxNormalizer parameters do not match"
    assert standard_normalizer.params == standard_normalizer_individual.params, "StandardNormalizer parameters do not match"

def test_meta_normalizer_forward_inverse(random_iterator, random_batch):
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = MetaNormalizer([minmax_normalizer, standard_normalizer])

    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    meta_normalizer.fit_params(iterator, axis=1, keys="input")

    # Test forward and inverse for MinMaxNormalizer
    normalized_minmax = minmax_normalizer.forward(random_batch, axis=1)
    denormalized_minmax = minmax_normalizer.inverse(normalized_minmax, axis=1)
    assert torch.allclose(random_batch, denormalized_minmax, atol=1e-6), "MinMaxNormalizer forward/inverse mismatch"

    # Test forward and inverse for StandardNormalizer
    normalized_standard = standard_normalizer.forward(random_batch, axis=1)
    denormalized_standard = standard_normalizer.inverse(normalized_standard, axis=1)
    assert torch.allclose(random_batch, denormalized_standard, atol=1e-6), "StandardNormalizer forward/inverse mismatch"

def test_meta_normalizer_different_keys(random_iterator):
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = MetaNormalizer([minmax_normalizer, standard_normalizer])

    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    meta_normalizer.fit_params(iterator, axis=1, keys=["input", "input"])

    # Ensure parameters are set correctly
    assert "x_min" in minmax_normalizer.params and "x_max" in minmax_normalizer.params, "MinMaxNormalizer parameters missing"
    assert "x_mean" in standard_normalizer.params and "x_mean_sq" in standard_normalizer.params, "StandardNormalizer parameters missing"

def test_save_and_load_meta_normalizer(tmp_path, random_iterator):
    # Create normalizers
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = MetaNormalizer([minmax_normalizer, standard_normalizer])

    # Fit the meta normalizer
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    meta_normalizer.fit_params(iterator, keys=["input", "input"], axis=1)

    # Save the meta normalizer
    file_names = ["minmax_normalizer.json", "standard_normalizer.json"]
    base_path = tmp_path / "meta_normalizer"
    meta_normalizer.save_as_json(file_names, base_path=str(base_path))

    # Load the normalizers individually
    loaded_minmax = MinMaxNormalizer.load_from_json(base_path / file_names[0])
    loaded_standard = StandardNormalizer.load_from_json(base_path / file_names[1])

    # Compare parameters
    assert minmax_normalizer.params == loaded_minmax.params, "MinMaxNormalizer parameters do not match after loading"
    assert standard_normalizer.params == loaded_standard.params, "StandardNormalizer parameters do not match after loading"
