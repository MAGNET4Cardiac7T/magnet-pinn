"""Test Blob real __init__ outside tests/generation/ autouse scope.

The tests/generation/conftest.py fast_blob_initialization fixture is
autouse=True, scope='module' and patches Blob.__init__ for all tests
discovered under tests/generation/. This file intentionally lives
outside that directory so the real __init__ runs.
"""

from unittest.mock import Mock

import numpy as np
import pytest

import magnet_pinn.generator.structures as structures
from magnet_pinn.generator.structures import Blob


def test_blob_real_init_sets_empirical_offsets():
    """Real Blob.__init__ computes empirical offsets and keeps the blob usable.

    The constructed instance must still execute real offset calculations.
    """
    position = np.array([0.0, 0.0, 0.0])
    radius = 0.1
    blob = Blob(position=position, radius=radius, num_octaves=1)
    sample_vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Real init must set all attributes from lines 154-168 of structures.py
    assert hasattr(blob, "empirical_max_offset")
    assert hasattr(blob, "empirical_min_offset")
    assert hasattr(blob, "effective_radius")
    assert hasattr(blob, "noise")
    assert blob.relative_disruption_strength == pytest.approx(0.1)
    assert blob.perlin_scale == pytest.approx(0.4)
    assert np.isclose(
        blob.effective_radius,
        radius * (1 + blob.empirical_max_offset),
    )
    assert blob.empirical_max_offset > blob.empirical_min_offset
    assert (
        blob.empirical_max_offset > 0
    ), "max offset must be positive so effective_radius >= radius"
    # Perlin noise spans both positive and negative values; with 10 000 sample
    # points the empirical minimum is reliably negative for the default setup.
    assert blob.empirical_min_offset < 0, "min offset must stay negative"

    offsets = blob.calculate_offsets(sample_vertices)

    assert offsets.shape == (sample_vertices.shape[0], 1)
    assert np.isfinite(offsets).all()
    assert np.all(
        offsets >= blob.empirical_min_offset
    ), "calculate_offsets result below empirical minimum"
    assert np.all(
        offsets <= blob.empirical_max_offset
    ), "calculate_offsets result above empirical maximum"


def test_blob_real_init_is_reproducible_for_fixed_seed():
    """Real Blob.__init__ is reproducible for a fixed seed."""
    position = np.array([0.0, 0.0, 0.0])
    radius = 0.1

    blob_a = Blob(position=position, radius=radius, num_octaves=1, seed=123)
    blob_b = Blob(position=position, radius=radius, num_octaves=1, seed=123)
    blob_c = Blob(position=position, radius=radius, num_octaves=1, seed=456)

    assert np.isclose(blob_a.empirical_max_offset, blob_b.empirical_max_offset)
    assert np.isclose(blob_a.empirical_min_offset, blob_b.empirical_min_offset)
    assert np.isclose(blob_a.effective_radius, blob_b.effective_radius)

    test_vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    offsets_a = blob_a.calculate_offsets(test_vertices)
    offsets_b = blob_b.calculate_offsets(test_vertices)
    assert np.allclose(
        offsets_a, offsets_b
    ), "identical seeds must produce identical calculate_offsets output"

    assert not (
        np.isclose(blob_a.empirical_max_offset, blob_c.empirical_max_offset)
        and np.isclose(
            blob_a.empirical_min_offset,
            blob_c.empirical_min_offset,
        )
        and np.isclose(blob_a.effective_radius, blob_c.effective_radius)
    ), "different seeds must change an init metric"


def test_blob_init_raises_on_zero_perlin_scale(
    monkeypatch: pytest.MonkeyPatch,
):
    """Blob.__init__ raises on `perlin_scale=0`
    before expensive setup starts.
    """
    position = np.array([0.0, 0.0, 0.0])
    perlin_noise_ctor = Mock(name="PerlinNoise")
    fibonacci_sampler = Mock(
        name="generate_fibonacci_points_on_sphere",
    )

    monkeypatch.setattr(structures, "PerlinNoise", perlin_noise_ctor)
    monkeypatch.setattr(
        structures,
        "generate_fibonacci_points_on_sphere",
        fibonacci_sampler,
    )

    with pytest.raises(ValueError, match="perlin_scale cannot be zero"):
        Blob(
            position=position,
            radius=0.1,
            num_octaves=1,
            perlin_scale=0,
        )

    perlin_noise_ctor.assert_not_called()
    fibonacci_sampler.assert_not_called()
