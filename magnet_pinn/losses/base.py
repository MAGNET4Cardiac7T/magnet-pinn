"""Base regression loss classes for neural network training."""

from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Tuple

from .utils import LossReducer


class BaseRegressionLoss(torch.nn.Module, ABC):
    """
    Base class for regression losses.
    """

    def __init__(
        self, feature_dims: Union[int, Tuple[int, ...]] = 1, reduction: str = "mean"
    ):
        """
        Initialize BaseRegressionLoss.

        Parameters
        ----------
        feature_dims : Union[int, Tuple[int, ...]], optional
            Dimensions over which to average the loss (default: 1)
        reduction : str, optional
            Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
        """
        super(BaseRegressionLoss, self).__init__()
        self.feature_dims = feature_dims

        if reduction is None or reduction == "none":
            self.reduction = lambda loss, mask: loss
        else:
            self.reduction = LossReducer(agg=reduction)

    @abstractmethod
    def _base_loss_fn(self, pred, target):
        """
        Compute the base loss function.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss tensor
        """
        raise NotImplementedError

    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        """
        Compute the forward pass of the loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values
        mask : Optional[torch.Tensor], optional
            Optional mask for selective loss computation (default: None)

        Returns
        -------
        torch.Tensor
            Reduced loss value
        """
        loss = self._base_loss_fn(pred, target)
        loss = torch.mean(loss, dim=self.feature_dims)
        return self.reduction(loss, mask)


class MSELoss(BaseRegressionLoss):
    """
    Mean Squared Error Loss

    .. math::

        L = \\frac{1}{n_{\\text{samples}}} \\sum_{i=1}^{n_{\\text{samples}}}
            (y_i - \\hat{y}_i)^2
    """

    def _base_loss_fn(self, pred, target):
        """
        Compute squared error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Squared error
        """
        return (pred - target) ** 2


class MAELoss(BaseRegressionLoss):
    """
    Mean Absolute Error Loss

    .. math::

        L = \\frac{1}{n_{\\text{samples}}} \\sum_{i=1}^{n_{\\text{samples}}}
            \\lvert y_i - \\hat{y}_i \\rvert
    """

    def _base_loss_fn(self, pred, target):
        """
        Compute absolute error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Absolute error
        """
        return torch.abs(pred - target)


class HuberLoss(BaseRegressionLoss):
    """
    Huber Loss
    """

    def __init__(
        self,
        delta: float = 1.0,
        feature_dims: Union[int, Tuple[int, ...]] = 1,
    ):
        """
        Initialize HuberLoss.

        Parameters
        ----------
        delta : float, optional
            Threshold for switching between quadratic and linear loss (default: 1.0)
        feature_dims : Union[int, Tuple[int, ...]], optional
            Dimensions over which to average the loss (default: 1)
        """
        super(HuberLoss, self).__init__(feature_dims=feature_dims)
        self.delta = delta

    def _base_loss_fn(self, pred, target):
        """
        Compute Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Huber loss (quadratic for small errors, linear for large errors)
        """
        loss = torch.abs(pred - target)
        return torch.where(
            loss < self.delta,
            0.5 * loss**2,
            self.delta * (loss - 0.5 * self.delta),
        )


class LogCoshLoss(BaseRegressionLoss):
    """
    Log-Cosh Loss

    .. math::

        L(y, \\hat{y}) = \\frac{1}{n_{\\text{samples}}}
            \\sum_{i=1}^{n_{\\text{samples}}}
            \\log\\left( \\cosh\\big( \\hat{y}_i - y_i \\big)
            \\right)
    """

    def _base_loss_fn(self, pred, target):
        """
        Compute log-cosh loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Log-cosh of the error
        """
        return torch.log(torch.cosh(pred - target))
