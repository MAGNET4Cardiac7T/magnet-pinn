from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Tuple

from .utils import MaskedLossReducer

class BaseRegressionLoss(torch.nn.Module, ABC):
    """
    Base class for regression losses.
    """
    def __init__(self, 
                 feature_dims: Union[int, Tuple[int, ...]] = 1):
        super(BaseRegressionLoss, self).__init__()
        self.feature_dims = feature_dims
        self.masked_reduction = MaskedLossReducer()

    @abstractmethod
    def _base_loss_fn(self, pred, target):
        raise NotImplementedError
        
    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        loss = self._base_loss_fn(pred, target)
        loss = torch.mean(loss, dim=self.feature_dims)
        return self.masked_reduction(loss, mask)

class MSELoss(BaseRegressionLoss):
    """
    Mean Squared Error Loss

    .. math::

        L = \\frac{1}{n_{\\text{samples}}} \\sum_{i=1}^{n_{\\text{samples}}}
            (y_i - \\hat{y}_i)^2
    """
    def _base_loss_fn(self, pred, target):
        return (pred - target) ** 2

class MAELoss(BaseRegressionLoss):
    """
    Mean Absolute Error Loss

    .. math::

        L = \\frac{1}{n_{\\text{samples}}} \\sum_{i=1}^{n_{\\text{samples}}}
            \\lvert y_i - \\hat{y}_i \\rvert
    """
    def _base_loss_fn(self, pred, target):
        return torch.abs(pred - target)    
    
class HuberLoss(BaseRegressionLoss):
    """
    Huber Loss
    """
    def __init__(self, delta: float = 1.0, feature_dims: Union[int, Tuple[int, ...]] = 1):
        super(HuberLoss, self).__init__(feature_dims=feature_dims)
        self.delta = delta

    def _base_loss_fn(self, pred, target):
        loss = torch.abs(pred - target)
        return torch.where(loss < self.delta, 0.5 * loss ** 2, self.delta * (loss - 0.5 * self.delta))
    
class LogCoshLoss(BaseRegressionLoss):
    """
    Log-Cosh Loss

    .. math::

        L(y, \\hat{y}) = \\frac{1}{n_{\\text{samples}}} \\sum_{i=1}^{n_{\\text{samples}}}
            \\log\\left( \\cosh\\big( \\hat{y}_i - y_i \\big) \\right)
    """
    def _base_loss_fn(self, pred, target):
        return torch.log(torch.cosh(pred - target))