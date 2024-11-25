from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Tuple

from .utils import MaskedLossReducer, partial_derivative_findiff

# TODO Add dx dy dz as parameters in a clever way
class BasePhysicsLoss(torch.nn.Module, ABC):
    def __init__(self, 
                 feature_dims: Union[int, Tuple[int, ...]] = 1):
        super(BasePhysicsLoss, self).__init__()
        self.feature_dims = feature_dims
        self.physics_filters = self._build_physics_filters()
        self.masked_reduction = MaskedLossReducer()

    @abstractmethod
    def _base_physics_fn(self, pred, target):
        raise NotImplementedError
    
    @abstractmethod
    def _build_physics_filters(self):
        raise NotImplementedError
        
    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        loss = self._base_physics_fn(pred, target)
        loss = torch.mean(loss, dim=self.feature_dims)
        return loss#self.masked_reduction(loss, mask)


class DivergenceLoss(BasePhysicsLoss):
    def _base_physics_fn(self, pred, target):
        return torch.nn.functional.conv3d(pred, self.physics_filters, padding=1)
    
    def _build_physics_filters(self):
        partial_derivative_findiff_coeffs = torch.tensor(partial_derivative_findiff(), dtype=torch.float32)
        divergence_filter = torch.zeros([1,3,3,3,3], dtype=torch.float32)
        divergence_filter[0,0,:,0,0] = partial_derivative_findiff_coeffs
        divergence_filter[0,1,0,:,0] = partial_derivative_findiff_coeffs
        divergence_filter[0,2,0,0,:] = partial_derivative_findiff_coeffs
        return divergence_filter