import pytest
import torch
from typing import Any, cast
from magnet_pinn.utils._normalization import (
    Identity,
    Power,
    Log,
    Tanh,
    Arcsinh,
    Nonlinearity,
    Normalizer,
    MinMaxNormalizer,
    StandardNormalizer,
    MetaNormalizer,
)


class DummyNormalizer(Normalizer):
    def _normalize(self, x):
        return Normalizer._normalize(self, x)

    def _denormalize(self, x):
        return Normalizer._denormalize(self, x)

    def _reset_params(self):
        return Normalizer._reset_params(self)

    def _update_params(self, x):
        return Normalizer._update_params(self, x)


class ConcreteMetaNormalizer(MetaNormalizer):
    def _update_params(self, x, axis: int = 0):
        return None


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


def test_minmax_normalizer_fit(random_iterator, random_batch):
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


def test_minmax_normalizer_correctness(random_iterator, random_batch):
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
    meta_normalizer = ConcreteMetaNormalizer([minmax_normalizer, standard_normalizer])

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
    assert (
        standard_normalizer.params == standard_normalizer_individual.params
    ), "StandardNormalizer parameters do not match"


def test_meta_normalizer_forward_inverse(random_iterator, random_batch):
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = ConcreteMetaNormalizer([minmax_normalizer, standard_normalizer])

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
    meta_normalizer = ConcreteMetaNormalizer([minmax_normalizer, standard_normalizer])

    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    meta_normalizer.fit_params(iterator, axis=1, keys=["input", "input"])

    # Ensure parameters are set correctly
    assert (
        "x_min" in minmax_normalizer.params and "x_max" in minmax_normalizer.params
    ), "MinMaxNormalizer parameters missing"
    assert (
        "x_mean" in standard_normalizer.params and "x_mean_sq" in standard_normalizer.params
    ), "StandardNormalizer parameters missing"


def test_save_and_load_meta_normalizer(tmp_path, random_iterator):
    # Create normalizers
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = ConcreteMetaNormalizer([minmax_normalizer, standard_normalizer])

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

@pytest.mark.parametrize("normalizer_class", [StandardNormalizer, MinMaxNormalizer])
@pytest.mark.parametrize("nonlinearity_class", [Identity, Power, Log, Tanh, Arcsinh])
@pytest.mark.parametrize("nonlinearity_before", [True, False])
def test_normalizer_with_nonlinearity(random_iterator, random_batch, normalizer_class, nonlinearity_class, nonlinearity_before):
    """Test that normalizers correctly apply nonlinearity transformations."""
    if nonlinearity_class == Power:
        nonlinearity = nonlinearity_class(power=2.0)
    else:
        nonlinearity = nonlinearity_class()

    normalizer = normalizer_class(nonlinearity=nonlinearity, nonlinearity_before=nonlinearity_before)
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    # Test forward and inverse
    normalized = normalizer.forward(random_batch, axis=1)
    denormalized = normalizer.inverse(normalized, axis=1)

    # Should reconstruct original data
    assert torch.allclose(random_batch, denormalized, atol=1e-5), \
        f"Failed to reconstruct with {normalizer_class.__name__}, {nonlinearity_class.__name__}, before={nonlinearity_before}"


def test_standard_normalizer_nonlinearity_after_normalization(random_iterator, random_batch):
    """Test that nonlinearity is applied after normalization when nonlinearity_before=False."""
    normalizer = StandardNormalizer(nonlinearity=Tanh(), nonlinearity_before=False)
    iterator = random_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    # Get normalized output
    normalized = normalizer.forward(random_batch, axis=1)

    # With Tanh applied after normalization, all values should be in [-1, 1]
    assert (normalized >= -1.0).all() and (normalized <= 1.0).all(), \
        "Tanh nonlinearity should bound values to [-1, 1]"

def test_standard_normalizer_nonlinearity_before_normalization(random_iterator):
    """Test that nonlinearity is applied before normalization when nonlinearity_before=True."""
    # Use positive data to test with Log nonlinearity
    def positive_iterator(seed=42, num_batches=10, batch_size=5, num_features=3):
        torch.manual_seed(seed)
        for _ in range(num_batches):
            yield {"input": torch.rand(batch_size, num_features) + 0.1}  # Ensure positive

    normalizer = StandardNormalizer(nonlinearity=Log(), nonlinearity_before=True)
    iterator = positive_iterator(seed=42, num_batches=10, batch_size=10, num_features=3)
    normalizer.fit_params(iterator, axis=1)

    # Test with positive data
    test_data = torch.rand(10, 3) + 0.1
    normalized = normalizer.forward(test_data, axis=1)
    denormalized = normalizer.inverse(normalized, axis=1)

    # Should reconstruct original data
    assert torch.allclose(test_data, denormalized, atol=1e-5), \
        "Failed to reconstruct with Log nonlinearity before normalization"


def test_nonlinearity_base_methods_raise():
    placeholder: Any = object()
    with pytest.raises(NotImplementedError):
        Nonlinearity.forward(placeholder, torch.tensor([1.0]))
    with pytest.raises(NotImplementedError):
        Nonlinearity.inverse(placeholder, torch.tensor([1.0]))


def test_normalizer_base_methods_raise():
    normalizer = DummyNormalizer()
    with pytest.raises(NotImplementedError):
        normalizer._normalize(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError):
        normalizer._denormalize(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError):
        normalizer._reset_params()
    with pytest.raises(NotImplementedError):
        normalizer._update_params(torch.tensor([1.0]))


def test_normalizer_helper_methods():
    normalizer = DummyNormalizer(params={"x_min": torch.tensor([0.0, 1.0])})
    axes = normalizer.get_reduction_axes(4, 1)
    assert axes == (0, 2, 3)

    expanded = normalizer._expand_params(axis=1, ndims=3)
    assert expanded["x_min"].shape == (1, 2, 1)


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("Identity", Identity),
        ("Power", Power),
        ("Log", Log),
        ("Tanh", Tanh),
        ("Arcsinh", Arcsinh),
    ],
)
def test_get_nonlinearity_function(name, expected_class):
    nonlinearity = Normalizer._get_nonlineartiy_function(name)
    assert isinstance(nonlinearity, expected_class)


def test_get_nonlinearity_function_invalid():
    with pytest.raises(ValueError):
        Normalizer._get_nonlineartiy_function("UnknownNonlinearity")


def test_meta_normalizer_not_implemented_methods():
    meta_normalizer = ConcreteMetaNormalizer([])
    with pytest.raises(NotImplementedError):
        meta_normalizer._normalize(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError):
        meta_normalizer._denormalize(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError):
        meta_normalizer._expand_params()
    with pytest.raises(NotImplementedError):
        meta_normalizer._cast_params()


def test_meta_normalizer_invalid_keys_length():
    minmax_normalizer = MinMaxNormalizer()
    standard_normalizer = StandardNormalizer()
    meta_normalizer = ConcreteMetaNormalizer([minmax_normalizer, standard_normalizer])

    with pytest.raises(ValueError):
        meta_normalizer.fit_params([{"input": torch.tensor([1.0])}], keys=["input"])


def test_meta_normalizer_invalid_keys_type():
    meta_normalizer = ConcreteMetaNormalizer([MinMaxNormalizer(), StandardNormalizer()])

    with pytest.raises(TypeError):
        meta_normalizer.fit_params([], keys=cast(Any, {"input": "value"}))


def test_meta_normalizer_save_validation_errors():
    meta_normalizer = ConcreteMetaNormalizer([MinMaxNormalizer(), StandardNormalizer()])

    with pytest.raises(ValueError):
        meta_normalizer.save_as_json(["only_one.json"])

    with pytest.raises(TypeError):
        meta_normalizer.save_as_json(["a.json", "b.json"], base_path=cast(Any, 123))
