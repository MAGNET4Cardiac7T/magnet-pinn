import pytest
import torch

from magnet_pinn.losses.base import MSELoss, MAELoss, HuberLoss, LogCoshLoss


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def feature_size():
    return 8


def test_mse_loss_basic(batch_size, feature_size):
    loss_fn = MSELoss()
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)

    loss = loss_fn(pred, target)
    expected = torch.mean((pred - target) ** 2)

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_mse_loss_with_mask(batch_size, feature_size):
    loss_fn = MSELoss()
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[: batch_size // 2] = False

    loss = loss_fn(pred, target, mask=mask)
    errors = torch.mean((pred - target) ** 2, dim=1)
    expected = torch.mean(errors[mask])

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_mse_loss_perfect_prediction():
    loss_fn = MSELoss()
    pred = torch.ones(5, 3)
    target = torch.ones(5, 3)

    loss = loss_fn(pred, target)

    assert loss.item() < 1e-6


def test_mae_loss_basic(batch_size, feature_size):
    loss_fn = MAELoss()
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)

    loss = loss_fn(pred, target)
    expected = torch.mean(torch.abs(pred - target))

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_mae_loss_with_mask(batch_size, feature_size):
    loss_fn = MAELoss()
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[: batch_size // 2] = False

    loss = loss_fn(pred, target, mask=mask)
    errors = torch.mean(torch.abs(pred - target), dim=1)
    expected = torch.mean(errors[mask])

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_mae_loss_perfect_prediction():
    loss_fn = MAELoss()
    pred = torch.ones(5, 3)
    target = torch.ones(5, 3)

    loss = loss_fn(pred, target)

    assert loss.item() < 1e-6


def test_huber_loss_basic(batch_size, feature_size):
    loss_fn = HuberLoss(delta=1.0)
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)

    loss = loss_fn(pred, target)
    delta = 1.0
    abs_diff = torch.abs(pred - target)
    expected = torch.mean(
        torch.where(
            abs_diff < delta,
            0.5 * abs_diff**2,
            delta * (abs_diff - 0.5 * delta),
        )
    )

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_huber_loss_small_errors():
    loss_fn = HuberLoss(delta=1.0)
    pred = torch.tensor([[0.1, 0.2, 0.3]])
    target = torch.tensor([[0.0, 0.0, 0.0]])

    loss = loss_fn(pred, target)
    expected = torch.mean(0.5 * (pred - target) ** 2)

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_huber_loss_large_errors():
    loss_fn = HuberLoss(delta=0.5)
    pred = torch.tensor([[2.0, 3.0, 4.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])

    loss = loss_fn(pred, target)
    delta = 0.5
    abs_diff = torch.abs(pred - target)
    expected = torch.mean(delta * (abs_diff - 0.5 * delta))

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_huber_loss_with_mask(batch_size, feature_size):
    loss_fn = HuberLoss(delta=1.0)
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[: batch_size // 2] = False

    loss = loss_fn(pred, target, mask=mask)
    delta = 1.0
    abs_diff = torch.abs(pred - target)
    errors = torch.mean(
        torch.where(
            abs_diff < delta,
            0.5 * abs_diff**2,
            delta * (abs_diff - 0.5 * delta),
        ),
        dim=1,
    )
    expected = torch.mean(errors[mask])

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_logcosh_loss_basic(batch_size, feature_size):
    loss_fn = LogCoshLoss()
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)

    loss = loss_fn(pred, target)
    expected = torch.mean(torch.log(torch.cosh(pred - target)))

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_logcosh_loss_with_mask(batch_size, feature_size):
    loss_fn = LogCoshLoss()
    pred = torch.randn(batch_size, feature_size)
    target = torch.randn(batch_size, feature_size)
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[: batch_size // 2] = False

    loss = loss_fn(pred, target, mask=mask)
    errors = torch.mean(torch.log(torch.cosh(pred - target)), dim=1)
    expected = torch.mean(errors[mask])

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)


def test_logcosh_loss_perfect_prediction():
    loss_fn = LogCoshLoss()
    pred = torch.ones(5, 3)
    target = torch.ones(5, 3)

    loss = loss_fn(pred, target)

    assert loss.item() < 1e-6


def test_loss_feature_dims_tuple():
    loss_fn = MSELoss(feature_dims=(1, 2))
    pred = torch.randn(4, 3, 5, 5)
    target = torch.randn(4, 3, 5, 5)

    loss = loss_fn(pred, target)
    expected = torch.mean((pred - target) ** 2, dim=(1, 2)).mean()

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-8)
