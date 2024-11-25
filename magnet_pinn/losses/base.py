from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Tuple

from .utils import MaskedLossReducer

class BaseRegressionLoss(torch.nn.Module, ABC):
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
    def _base_loss_fn(self, pred, target):
        return (pred - target) ** 2

class MAELoss(BaseRegressionLoss):
    def _base_loss_fn(self, pred, target):
        return torch.abs(pred - target)    
    
class HuberLoss(BaseRegressionLoss):
    def __init__(self, delta: float = 1.0, feature_dims: Union[int, Tuple[int, ...]] = 1):
        super(HuberLoss, self).__init__(feature_dims=feature_dims)
        self.delta = delta

    def _base_loss_fn(self, pred, target):
        loss = torch.abs(pred - target)
        return torch.where(loss < self.delta, 0.5 * loss ** 2, self.delta * (loss - 0.5 * self.delta))
    
class LogCoshLoss(BaseRegressionLoss):
    def _base_loss_fn(self, pred, target):
        return torch.log(torch.cosh(pred - target))