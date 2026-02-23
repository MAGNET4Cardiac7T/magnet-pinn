import pytest
import torch
import math

from magnet_pinn.losses.physics import DivergenceLoss, FaradaysLawLoss, MRI_FREQUENCY_HZ, VACUUM_PERMEABILITY
from magnet_pinn.losses import MRI_FREQUENCY_HZ as MRI_FREQ_EXPORTED, VACUUM_PERMEABILITY as VACUUM_PERM_EXPORTED


@pytest.fixture
def spatial_size():
    return 16


@pytest.fixture
def batch_size():
    return 2


def test_divergence_loss_shape(batch_size, spatial_size):
    loss_fn = DivergenceLoss()
    pred = torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size)
    mask = None

    loss = loss_fn(pred, mask)

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_divergence_loss_with_mask(batch_size, spatial_size):
    loss_fn = DivergenceLoss()
    pred = torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size)
    target = torch.zeros_like(pred)
    mask = torch.ones(batch_size, spatial_size, spatial_size, spatial_size, dtype=torch.bool)
    mask[:, :spatial_size//2] = False

    loss = loss_fn(pred, target, mask=mask)

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_divergence_loss_on_solenoidal_field():
    """
    Test that divergence loss is low for a solenoidal (zero-divergence) field.
    Using field V = (y, -x, 0) which has div(V) = 0.
    """
    loss_fn = DivergenceLoss()
    spatial_size = 32
    batch_size = 1

    x = torch.linspace(-1, 1, spatial_size)
    y = torch.linspace(-1, 1, spatial_size)
    z = torch.linspace(-1, 1, spatial_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    pred = torch.zeros(batch_size, 3, spatial_size, spatial_size, spatial_size)
    pred[0, 0] = Y
    pred[0, 1] = -X
    pred[0, 2] = 0

    target = torch.zeros_like(pred)
    loss = loss_fn(pred, target)

    dx = 2.0 / (spatial_size - 1)
    max_expected_error = 10 * dx**2

    assert loss.item() < max_expected_error, f"Loss should be near zero for solenoidal field, got {loss.item()}"


def test_faradays_loss_shape(batch_size, spatial_size, random_fields):
    loss_fn = FaradaysLawLoss()

    loss = loss_fn(random_fields)

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_faradays_loss_non_zero_for_violating_fields(violating_fields):
    """
    Test that Faraday's loss is non-zero for fields that violate Faraday's law.
    Using constant fields which don't satisfy curl(E) + jωμH = 0.
    """
    loss_fn = FaradaysLawLoss()

    loss = loss_fn(violating_fields)

    assert loss.item() > 1e-3, (
        f"Loss should be non-zero for fields violating Faraday's law, got {loss.item()}"
    )


def test_faradays_loss_with_mask(batch_size, spatial_size, random_fields, half_mask):
    loss_fn = FaradaysLawLoss()

    loss = loss_fn(random_fields, mask=half_mask)

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_faradays_loss_zero_for_trivial_zero_field(zero_fields):
    """
    Test that zero fields give zero loss (trivial case).
    """
    loss_fn = FaradaysLawLoss()

    loss = loss_fn(zero_fields)

    assert loss.item() < 1e-5, f"Loss should be near zero for zero fields, got {loss.item()}"


def test_faradays_loss_device_dtype_casting(random_fields, random_fields_float64):
    loss_fn = FaradaysLawLoss()
    loss = loss_fn(random_fields)
    assert loss.dtype == torch.float32


    loss = loss_fn(random_fields_float64)
    assert loss.dtype == torch.float64


def test_divergence_loss_on_non_zero_divergence_field():
    """
    Test that divergence loss is non-zero for a field with known divergence.
    Using field V = (x, y, z) which has div(V) = 3.
    """
    loss_fn = DivergenceLoss()
    spatial_size = 16
    batch_size = 1

    x = torch.linspace(-1, 1, spatial_size)
    y = torch.linspace(-1, 1, spatial_size)
    z = torch.linspace(-1, 1, spatial_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    pred = torch.zeros(batch_size, 3, spatial_size, spatial_size, spatial_size)
    pred[0, 0] = X
    pred[0, 1] = Y
    pred[0, 2] = Z

    target = torch.zeros_like(pred)
    loss = loss_fn(pred, target)

    assert loss.item() > 0.1, f"Loss should be significant for divergent field, got {loss.item()}"
    assert loss.item() < 10.0, f"Loss should be reasonable magnitude, got {loss.item()}"


def test_divergence_loss_gradient_flow():
    """
    Test that gradients flow correctly through divergence loss.
    """
    loss_fn = DivergenceLoss()
    spatial_size = 8
    batch_size = 1

    pred = torch.randn(
        batch_size, 3, spatial_size, spatial_size, spatial_size,
        dtype=torch.float32, requires_grad=True
    )
    target = torch.zeros_like(pred)

    loss = loss_fn(pred, target)
    loss.backward()

    assert pred.grad is not None, "Gradients should flow through divergence loss"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(pred.grad).any(), "Gradients should not contain Inf"


def test_faradays_loss_gradient_flow(random_fields_with_gradient):
    """
    Test that gradients flow correctly through Faraday's loss.
    """
    loss_fn = FaradaysLawLoss()

    loss = loss_fn(random_fields_with_gradient)
    loss.backward()

    for field in random_fields_with_gradient:
        assert field.grad is not None, "Gradients should flow through Faraday's loss"
        assert not torch.isnan(field.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(field.grad).any(), "Gradients should not contain Inf"


def test_divergence_loss_with_nan_input():
    """
    Test that divergence loss handles NaN inputs gracefully.
    """
    loss_fn = DivergenceLoss()
    spatial_size = 8
    batch_size = 1

    pred = torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size)
    pred[0, 0, 0, 0, 0] = float('nan')
    target = torch.zeros_like(pred)

    loss = loss_fn(pred, target)

    assert torch.isnan(loss), "Loss should be NaN when input contains NaN"


def test_divergence_loss_batch_consistency():
    """
    Test that divergence loss gives consistent results across batch dimension.
    """
    loss_fn = DivergenceLoss()
    spatial_size = 8

    single_sample = torch.randn(1, 3, spatial_size, spatial_size, spatial_size)
    pred = single_sample.repeat(4, 1, 1, 1, 1)
    target = torch.zeros_like(pred)

    losses = []
    for i in range(4):
        mask = torch.zeros(4, spatial_size, spatial_size, spatial_size, dtype=torch.bool)
        mask[i] = True
        loss = loss_fn(pred, target, mask=mask)
        losses.append(loss.item())

    losses_tensor = torch.tensor(losses)
    assert torch.std(losses_tensor).item() < 1e-5, (
        f"Losses should be identical for identical samples, got {losses}"
    )


def test_faradays_loss_physics_constants():
    """
    Test that Faraday's loss uses correct physics constants for 7T MRI.

    The constants should match:
    - MRI_FREQUENCY_HZ = 297.2e6 Hz (Larmor frequency at 7T for hydrogen)
    - VACUUM_PERMEABILITY = 1.256637061e-6 H/m (μ₀ = 4π × 10⁻⁷ H/m)
    """
    assert 295e6 < MRI_FREQUENCY_HZ < 300e6, (
        f"MRI frequency {MRI_FREQUENCY_HZ/1e6:.1f} MHz should be near 298 MHz for 7T"
    )

    mu_0_exact = 4 * math.pi * 1e-7
    assert abs(VACUUM_PERMEABILITY - mu_0_exact) < 1e-15, (
        f"Vacuum permeability should be 4π × 10⁻⁷ H/m, got {VACUUM_PERMEABILITY}"
    )

    assert MRI_FREQUENCY_HZ == 297.2e6, "MRI frequency should be 297.2 MHz"
    assert abs(VACUUM_PERMEABILITY - 1.256637061e-6) < 1e-15, (
        "Vacuum permeability should be 1.256637061e-6 H/m"
    )


def test_faradays_loss_angular_frequency_calculation():
    """
    Test that Faraday's loss correctly implements the angular frequency term.

    The term j·ω·μ·H should use ω = 2πf, where f is the MRI operating frequency.
    """
    omega = 2 * math.pi * MRI_FREQUENCY_HZ
    expected_coefficient = omega * VACUUM_PERMEABILITY

    assert 2300 < expected_coefficient < 2400, (
        f"Angular frequency coefficient ω·μ should be ~2347, got {expected_coefficient}"
    )

    calculated = 2 * math.pi * MRI_FREQUENCY_HZ * VACUUM_PERMEABILITY
    assert abs(calculated - expected_coefficient) < 1e-10


def test_physics_constants_are_exported():
    """
    Test that physics constants are properly exported from the losses module.
    """
    assert MRI_FREQ_EXPORTED == 297.2e6
    assert abs(VACUUM_PERM_EXPORTED - 1.256637061e-6) < 1e-15


def test_base_physics_loss_accuracy_passed_to_filter():
    """
    Test that the accuracy parameter is forwarded to DiffFilterFactory.
    """
    loss_fn_acc2 = DivergenceLoss(accuracy=2)
    loss_fn_acc4 = DivergenceLoss(accuracy=4)

    assert loss_fn_acc2.diff_filter_factory.accuracy == 2
    assert loss_fn_acc4.diff_filter_factory.accuracy == 4
    # Higher accuracy means larger filter kernel
    assert loss_fn_acc4.physics_filters.shape[-1] > loss_fn_acc2.physics_filters.shape[-1]


def test_base_physics_loss_invalid_dx_unit():
    """
    Test that an unrecognised dx_unit raises ValueError.
    """
    with pytest.raises(ValueError, match="dx_unit"):
        DivergenceLoss(dx_unit="km")


def test_base_physics_loss_dx_unit_stored():
    """
    Test that dx_unit and coordinate_scale are stored on the base class.
    """
    loss_m  = DivergenceLoss(dx=0.004, dx_unit="m")
    loss_mm = DivergenceLoss(dx=4.0,   dx_unit="mm")

    assert loss_m.dx_unit == "m"
    assert loss_m.coordinate_scale == 1.0
    assert loss_mm.dx_unit == "mm"
    assert abs(loss_mm.coordinate_scale - 1e-3) < 1e-15


def test_faradays_loss_omega_mu_scaling():
    """
    Test that _omega_mu is scaled correctly for different dx_unit values.
    ωμ₀ in mm coords should equal ωμ₀ in m coords multiplied by 1e-3 (mm/m).
    """
    loss_m  = FaradaysLawLoss(dx_unit="m")
    loss_mm = FaradaysLawLoss(dx_unit="mm")

    assert abs(loss_mm._omega_mu - loss_m._omega_mu * 1e-3) < 1e-20


def test_faradays_loss_dx_unit_mm_equivalent_to_si(random_fields):
    """
    Test that FaradaysLawLoss(dx=4.0, dx_unit='mm') and FaradaysLawLoss(dx=0.004,
    dx_unit='m') represent the same physics on a 4 mm grid.

    Both configurations scale all terms in the Faraday residual by the same factor
    (1e-3 for mm vs m), so the squared loss differs by scale**2 = 1e-6.
    The ratio loss_m / loss_mm should equal (1e3)**2 = 1e6.
    """
    loss_fn_m  = FaradaysLawLoss(dx=0.004, dx_unit="m")
    loss_fn_mm = FaradaysLawLoss(dx=4.0,   dx_unit="mm")

    loss_m  = loss_fn_m(random_fields)
    loss_mm = loss_fn_mm(random_fields)

    expected_ratio = (1e3) ** 2  # (m_per_mm)^2 because loss is squared residual
    actual_ratio = loss_m.item() / loss_mm.item()
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-4, (
        f"Loss ratio should be {expected_ratio:.2e}, got {actual_ratio:.6e} "
        f"(m={loss_m.item():.6e}, mm={loss_mm.item():.6e})"
    )
