import pytest
import torch
from magnet_pinn.losses.physics import DivergenceLoss, FaradaysLoss


@pytest.fixture
def spatial_size():
    return 16


@pytest.fixture
def batch_size():
    return 2


def test_divergence_loss_shape(batch_size, spatial_size):
    loss_fn = DivergenceLoss()
    pred = torch.randn(batch_size, 3, spatial_size, spatial_size, spatial_size)
    target = torch.zeros_like(pred)

    loss = loss_fn(pred, target)

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

    assert loss.item() < 0.02, f"Loss should be near zero for solenoidal field, got {loss.item()}"


def test_faradays_loss_shape(batch_size, spatial_size):
    loss_fn = FaradaysLoss()
    pred = torch.randn(batch_size, 12, spatial_size, spatial_size, spatial_size)
    target = torch.zeros_like(pred)

    loss = loss_fn(pred, target)

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_faradays_loss_non_zero_for_violating_fields():
    """
    Test that Faraday's loss is non-zero for fields that violate Faraday's law.
    Using constant fields which don't satisfy curl(E) + jωμH = 0.
    """
    loss_fn = FaradaysLoss()
    batch_size = 1
    spatial_size = 16

    pred = torch.zeros(batch_size, 12, spatial_size, spatial_size, spatial_size)

    pred[:, 0:3, :, :, :] = 1.0
    pred[:, 3:6, :, :, :] = 0.5

    pred[:, 6:9, :, :, :] = 0.1
    pred[:, 9:12, :, :, :] = 0.05

    target = torch.zeros_like(pred)
    loss = loss_fn(pred, target)

    assert loss.item() > 1e-3, (
        f"Loss should be non-zero for fields violating Faraday's law, got {loss.item()}"
    )


def test_faradays_loss_with_mask(batch_size, spatial_size):
    loss_fn = FaradaysLoss()
    pred = torch.randn(batch_size, 12, spatial_size, spatial_size, spatial_size)
    target = torch.zeros_like(pred)
    mask = torch.ones(batch_size, spatial_size, spatial_size, spatial_size, dtype=torch.bool)
    mask[:, :spatial_size//2] = False

    loss = loss_fn(pred, target, mask=mask)

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_faradays_loss_zero_for_satisfying_fields():
    loss_fn = FaradaysLoss()
    spatial_size = 16
    batch_size = 1

    pred = torch.zeros(batch_size, 12, spatial_size, spatial_size, spatial_size)
    target = torch.zeros_like(pred)

    loss = loss_fn(pred, target)

    assert loss.item() < 1e-5, f"Loss should be near zero for satisfying fields, got {loss.item()}"


def test_faradays_loss_device_dtype_casting():
    loss_fn = FaradaysLoss()
    spatial_size = 8
    batch_size = 1

    pred = torch.randn(batch_size, 12, spatial_size, spatial_size, spatial_size, dtype=torch.float32)
    target = torch.zeros_like(pred)
    loss = loss_fn(pred, target)
    assert loss.dtype == torch.float32

    pred = torch.randn(batch_size, 12, spatial_size, spatial_size, spatial_size, dtype=torch.float64)
    target = torch.zeros_like(pred)
    loss = loss_fn(pred, target)
    assert loss.dtype == torch.float64
