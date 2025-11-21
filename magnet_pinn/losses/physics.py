from abc import ABC, abstractmethod
import torch
from torch.nn import Identity
import math
from einops import pack, unpack
from typing import Optional, Union, Tuple

from .utils import LossReducer, DiffFilterFactory, ObjectMaskCropping


MRI_FREQUENCY_HZ = 297.2e6
VACUUM_PERMEABILITY = 1.256637061e-6


# TODO Add support for non-uniform grids (i.e., varying voxel sizes in different dimensions)
class BasePhysicsLoss(torch.nn.Module, ABC):
    """
    Base class for physics-based losses
    """
    def __init__(self, 
                 feature_dims: Union[int, Tuple[int, ...]] = 1,
                 reduction: Union[str, LossReducer, None] = None,
                 dx: float = 1.0
                 ):
        super(BasePhysicsLoss, self).__init__()
        self.feature_dims = feature_dims
        self.dx = dx
        
        if type(reduction) == str:
            if reduction == 'masked':
                self.reduction = LossReducer(agg='mean', masking=True)
            elif reduction == "full":
                self.reduction = LossReducer(agg='mean', masking=False)
            else:
                raise ValueError(f"Unknown reduction type: {reduction}")
        elif reduction is None:
            self.reduction = lambda loss, mask: loss
        else:
            self.reduction = reduction

        self.diff_filter_factory = DiffFilterFactory(dx = self.dx)

        self.physics_filters = self._build_physics_filters()

    @abstractmethod
    def _base_physics_fn(self, pred, target):
        raise NotImplementedError

    @abstractmethod
    def _build_physics_filters(self):
        raise NotImplementedError

    def _cast_physics_filter(
        self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')
    ) -> None:
        if self.physics_filters.dtype != dtype or self.physics_filters.device != device:
            self.physics_filters = self.physics_filters.to(dtype=dtype, device=device)

    def forward(self, field, mask: Optional[torch.Tensor] = None):
        self._cast_physics_filter(field.dtype, field.device)
        loss = self._base_physics_fn(field)
        loss = torch.mean(loss, dim=self.feature_dims)
        return self.reduction(loss, mask)


# TODO Add different Lp norms for the divergence residual
class DivergenceLoss(BasePhysicsLoss):
    """
    Divergence loss for calculating the divergence of a physical field. Used to enforce
    Gauss's Law and Gauss's Law for Magnetism for electromagnetic field prediction.
    
    Computes the residual of the divergence of a field 
    to enforce physical consistency of the predicted fields.
    
    .. math::
        \\nabla \\cdot \\mathbf{E} = \\frac{\\ro}{\\epsilon_0} \\\\
        \\nabla \\cdot \\mathbf{B} = 0
        
    In the forward pass, expects the predicted fields to be of shape
    (b, 3, x, y, z) or (3, x, y, z).
    
    The optional mask should be of shape
    (b, x, y, z) or (x, y, z) respectively.
        
    Parameters
    ----------
    feature_dims : Union[int, Tuple[int, ...]], optional
        Dimensions over which to average the loss before reduction,
        by default 1.
        
    Returns
    -------
    torch.Tensor
        Squared magnitude of the divergence of the predicted fields.
        With the spatial dimensions reduced if reduction is not None.
        
    """
    def _base_physics_fn(self, field):
        padding = self.diff_filter_factory.accuracy // 2
        divergence = torch.nn.functional.conv3d(
            field, self.physics_filters, padding=padding
        )
        return divergence**2

    def _build_physics_filters(self):
        divergence_filter = self.diff_filter_factory.divergence()
        return divergence_filter


class FaradayLawLoss(BasePhysicsLoss):
    """
    Faraday's Law Loss for electromagnetic field prediction.

    Computes the residual of Faraday's law in the frequency domain to
    enforce physical consistency between electric and magnetic fields.

    .. math::

        \\nabla \\times \\mathbf{E} + j\\omega\\mu\\mathbf{H} = 0
        
    In the forward pass, expects the predicted fields to be of shape
    (b, 2, 2, 3, x, y, z), where the b is the batch dimension.
    
    The 2, 2, 3 correspond to (E/H), (real/imaginary), and (x/y/z) components, respectively.

    Parameters
    ----------
    feature_dims : Union[int, Tuple[int, ...]], optional
        Dimensions over which to average the loss before reduction,
        by default 1.
    """
    def _base_physics_fn(self, pred, target):
        """
        Compute the Faraday's law residual.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted electromagnetic fields with shape
            (batch, 12, x, y, z).
        target : torch.Tensor
            Target fields (unused in physics loss computation).

        Returns
        -------
        torch.Tensor
            Squared magnitude of the Faraday's law residual.
        """
        pred_e_re = pred[:, 0:3, :, :, :]
        pred_e_im = pred[:, 3:6, :, :, :]

        padding = self.diff_filter_factory.accuracy // 2
        curl_pred_e_re = torch.nn.functional.conv3d(
            pred_e_re, self.physics_filters, padding=padding
        )
        curl_pred_e_im = torch.nn.functional.conv3d(
            pred_e_im, self.physics_filters, padding=padding
        )

        curl_pred_e = curl_pred_e_re + 1j * curl_pred_e_im

        pred_h_re = pred[:, 6:9, :, :, :]
        pred_h_im = pred[:, 9:12, :, :, :]

        pred_h = pred_h_re + 1j * pred_h_im

        faradays_pred = (
            curl_pred_e
            + 1j * 2 * math.pi * MRI_FREQUENCY_HZ * VACUUM_PERMEABILITY * pred_h
        )

        return faradays_pred.abs() ** 2

    def _build_physics_filters(self):
        """
        Build the curl operator filter.

        Returns
        -------
        torch.Tensor
            Curl operator filter.
        """
        curl_filter = self.diff_filter_factory.curl()
        return curl_filter
