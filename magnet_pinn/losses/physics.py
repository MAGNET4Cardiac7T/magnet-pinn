"""Physics-informed loss functions for electromagnetic field prediction."""
from abc import ABC, abstractmethod
import torch
from torch.nn import Identity
import math
from einops import pack, unpack
from typing import Optional, Union, Tuple

from .utils import LossReducer, DiffFilterFactory, ObjectMaskCropping, ResidualNorm
from .base import BaseRegressionLoss


MRI_FREQUENCY_HZ = 297.2e6
VACUUM_PERMEABILITY = 1.256637061e-6

COORDINATE_UNIT_SCALES = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}


# TODO Add support for non-uniform grids (i.e., varying voxel sizes in different dimensions)
class BasePhysicsLoss(BaseRegressionLoss):
    """
    Base class for physics-based losses
    """
    def __init__(self,
                 feature_dims: Union[int, Tuple[int, ...]] = 1,
                 reduction: str = "mean",
                 dx: float = 1.0,
                 dx_unit: str = "m",
                 accuracy: int = 2,
                 residual_norm: str = "l2",
                 p: float = 2.0,
                 ):
        """
        Initialize BasePhysicsLoss.

        Parameters
        ----------
        feature_dims : Union[int, Tuple[int, ...]], optional
            Dimensions over which to average the loss (default: 1)
        reduction : str, optional
            Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
        dx : float, optional
            Grid spacing for finite difference calculations (default: 1.0)
        dx_unit : str, optional
            Unit of the grid spacing. One of 'm', 'cm', 'mm' (default: 'm').
            Subclasses that contain physical constants with SI units use this
            to keep those constants dimensionally consistent with the chosen
            coordinate unit.
        accuracy : int, optional
            Order of accuracy for the finite difference approximation (default: 2)
        residual_norm : str, optional
            Norm applied to the physics residual magnitude ``|r|`` before
            reduction. One of:

            * ``"l2"``   – squared magnitude ``|r|²`` (default, original behaviour)
            * ``"l1"``   – absolute magnitude ``|r|``
            * ``"lp"``   – ``|r|^p``  (use ``p`` to set the exponent)
            * ``"rmse"`` – element-wise ``|r|²``, with ``sqrt`` applied to the
              final scalar after reduction

        p : float, optional
            Exponent for ``residual_norm="lp"``. Must be positive (default: 2.0)
        """
        if dx_unit not in COORDINATE_UNIT_SCALES:
            raise ValueError(
                f"dx_unit must be one of {list(COORDINATE_UNIT_SCALES)}, got '{dx_unit}'"
            )
        super(BasePhysicsLoss, self).__init__(feature_dims=feature_dims, reduction=reduction)

        self.dx = dx
        self.dx_unit = dx_unit
        self.coordinate_scale = COORDINATE_UNIT_SCALES[dx_unit]
        self.diff_filter_factory = DiffFilterFactory(dx=self.dx, accuracy=accuracy)
        self.physics_filters = self._build_physics_filters()
        self._residual_norm = ResidualNorm(norm=residual_norm, p=p)
        self._apply_sqrt_after_reduce = (residual_norm == "rmse")

    @abstractmethod
    def _base_physics_fn(self,
                         field: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics-based residual for the field.

        Parameters
        ----------
        field : torch.Tensor
            Input field tensor

        Returns
        -------
        torch.Tensor
            Physics residual
        """
        raise NotImplementedError

    @abstractmethod
    def _build_physics_filters(self):
        """
        Build physics-specific differential operator filters.

        Returns
        -------
        torch.Tensor
            Differential operator filter
        """
        raise NotImplementedError

    def _base_loss_fn(self, pred, target):
        """
        Compute the base physics loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted field
        target : torch.Tensor or None
            Target field (if None, assumes zero residual)

        Returns
        -------
        torch.Tensor
            Element-wise residual loss according to ``residual_norm``
        """
        dtype, device = self._check_dtype_device(pred)
        self._cast_physics_filter(dtype, device)

        residual_pred = self._base_physics_fn(pred)
        if target is not None:
            residual_target = self._base_physics_fn(target)
        else:
            residual_target = torch.zeros_like(residual_pred)

        residual_magnitude = (residual_pred - residual_target).abs()
        loss = self._residual_norm(residual_magnitude)
        return loss

    def _cast_physics_filter(
        self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')
    ) -> None:
        """
        Cast physics filters to specified dtype and device.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Target data type (default: torch.float32)
        device : torch.device, optional
            Target device (default: cpu)
        """
        if self.physics_filters.dtype != dtype or self.physics_filters.device != device:
            self.physics_filters = self.physics_filters.to(dtype=dtype, device=device)

    def _check_dtype_device(self, field: torch.Tensor) -> Tuple[torch.dtype, torch.device]:
        """
        Extract dtype and device from field tensor.

        Parameters
        ----------
        field : torch.Tensor or collection of torch.Tensor
            Input field(s)

        Returns
        -------
        Tuple[torch.dtype, torch.device]
            Data type and device of the field

        Raises
        ------
        ValueError
            If field is not a tensor or collection of tensors
        """
        if isinstance(field, torch.Tensor):
            return field.dtype, field.device
        elif isinstance(field, (list, tuple)):
            return field[0].dtype, field[0].device
        elif isinstance(field, dict):
            first_value = next(iter(field.values()))
            return first_value.dtype, first_value.device
        else:
            raise ValueError("Input field must be a torch.Tensor or a collection of torch.Tensors.")

    def forward(self,
                pred,
                target: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        """
        Compute the forward pass of the physics loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted field
        target : Optional[torch.Tensor], optional
            Target field (default: None)
        mask : Optional[torch.Tensor], optional
            Mask for selective loss computation (default: None)

        Returns
        -------
        torch.Tensor
            Reduced physics loss. When ``residual_norm="rmse"``, the square
            root is applied to the scalar after reduction.
        """
        loss = super().forward(pred, target, mask)
        if self._apply_sqrt_after_reduce:
            loss = torch.sqrt(loss)
        return loss


class DivergenceLoss(BasePhysicsLoss):
    """
    Divergence loss for calculating the divergence of a physical field. Used to enforce
    Gauss's Law and Gauss's Law for Magnetism for electromagnetic field prediction.

    Computes the residual of the divergence of a field
    to enforce physical consistency of the predicted fields.

    .. math::
        \\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0} \\\\
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
        """
        Compute the divergence of the field.

        Parameters
        ----------
        field : torch.Tensor
            Input vector field of shape (b, 3, x, y, z)

        Returns
        -------
        torch.Tensor
            Divergence of the field
        """
        padding = self.diff_filter_factory.accuracy // 2
        divergence = torch.nn.functional.conv3d(
            field, self.physics_filters, padding=padding
        )
        return divergence

    def _build_physics_filters(self):
        """
        Build divergence operator filter.

        Returns
        -------
        torch.Tensor
            Divergence operator filter
        """
        divergence_filter = self.diff_filter_factory.divergence()
        return divergence_filter


class FaradaysLawLoss(BasePhysicsLoss):
    """
    Faraday's Law Loss for electromagnetic field prediction.

    Computes the residual of Faraday's law in the frequency domain to
    enforce physical consistency between electric and magnetic fields.

    .. math::

        \\nabla \\times \\mathbf{E} + j\\omega\\mu\\mathbf{H} = 0

    In the forward pass, expects a tuple of predicted fields:
    (efield_real, efield_imag, hfield_real, hfield_imag),
    each of shape (b, 3, x, y, z) or (3, x, y, z).

    The 2, 2, 3 correspond to (E/H), (real/imaginary), and (x/y/z) components, respectively.

    Parameters
    ----------
    feature_dims : Union[int, Tuple[int, ...]], optional
        Dimensions over which to average the loss before reduction,
        by default 1.
    dx_unit : str, optional
        Unit of the grid spacing. One of 'm', 'cm', 'mm' (default: 'm').
        Scales the ωμ₀ constant so that the Faraday residual remains
        dimensionally consistent when coordinates are not in SI metres.
    """
    vacuum_permeability: float = VACUUM_PERMEABILITY
    mri_frequency_hz: float = MRI_FREQUENCY_HZ

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._omega_mu = (
            2 * math.pi * self.mri_frequency_hz * self.vacuum_permeability
            * self.coordinate_scale
        )

    def _base_physics_fn(self, field):
        """
        Compute the Faraday's law residual.

        Parameters
        ----------
        pred : torch.Tensor
            Tuple of predicted fields (efield_real, efield_imag, hfield_real, hfield_imag).
            Each tensor should have shape (b, 3, x, y, z) or (3, x, y, z).
        target : torch.Tensor
            Target fields (unused in physics loss computation).

        Returns
        -------
        torch.Tensor
            Squared magnitude of the Faraday's law residual.
        """
        pred_e_re = field[0]
        pred_e_im = field[1]
        pred_h_re = field[2]
        pred_h_im = field[3]

        padding = self.diff_filter_factory.accuracy // 2
        curl_pred_e_re = torch.nn.functional.conv3d(
            pred_e_re, self.physics_filters, padding=padding
        )
        curl_pred_e_im = torch.nn.functional.conv3d(
            pred_e_im, self.physics_filters, padding=padding
        )

        curl_pred_e = curl_pred_e_re + 1j * curl_pred_e_im
        pred_h = pred_h_re + 1j * pred_h_im

        faradays_pred = (
            curl_pred_e
            + 1j * self._omega_mu * pred_h
        )

        return faradays_pred

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
