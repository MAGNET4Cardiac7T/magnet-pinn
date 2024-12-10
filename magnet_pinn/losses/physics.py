from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Tuple

from .utils import MaskedLossReducer, DiffFilterFactory, ObjectMaskPadding

# TODO Add dx dy dz as parameters in a clever way
class BasePhysicsLoss(torch.nn.Module, ABC):
    def __init__(self, 
                 feature_dims: Union[int, Tuple[int, ...]] = 1):
        super(BasePhysicsLoss, self).__init__()
        self.feature_dims = feature_dims
        self.masked_reduction = MaskedLossReducer()
        self.diff_filter_factory = DiffFilterFactory()
        self.object_mask_padding = ObjectMaskPadding()
        
        self.physics_filters = self._build_physics_filters()

    @abstractmethod
    def _base_physics_fn(self, pred, target):
        raise NotImplementedError
    
    @abstractmethod
    def _build_physics_filters(self):
        raise NotImplementedError
    
    def _cast_physics_filter(self, 
                             dtype: torch.dtype = torch.float32, 
                             device: torch.device = torch.device('cpu')) -> None:
        if self.physics_filters.dtype != dtype or self.physics_filters.device != device:
            self.physics_filters = self.physics_filters.to(dtype=dtype, device=device)
        
    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        self._cast_physics_filter(pred.dtype, pred.device)
        loss = self._base_physics_fn(pred, target)
        loss = torch.mean(loss, dim=self.feature_dims)
        return self.masked_reduction(loss, mask)

# TODO Add different Lp norms for the divergence residual
# TODO Calculate padding based on accuracy of the finite difference filter
class DivergenceLoss(BasePhysicsLoss):
    def _base_physics_fn(self, pred, target):
        return torch.nn.functional.conv3d(pred, self.physics_filters, padding=1)**2
    
    def _build_physics_filters(self):
        divergence_filter = self.diff_filter_factory.divergence()
        return divergence_filter