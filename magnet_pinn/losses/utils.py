import torch
from typing import Optional
from findiff import coefficients
import einops
from operator import mul
from functools import reduce


class LossReducer(torch.nn.Module): 
    """
    Loss reducer with optional masking. It reduces the loss tensor using the specified aggregation function (mean, sum, or none).
    Masking can be applied to consider only specific elements of the loss tensor during reduction.
    
    Parameters
    ----------
    agg : str, optional
        Aggregation method to use: 'mean', 'sum', or 'none'. Default is 'mean'.
    masking : bool, optional
        Whether to apply masking during reduction. Default is True.
    """
    
    def __init__(self,
                 agg: str = 'mean', 
                 masking: bool = True):
        super(LossReducer, self).__init__()
        if not agg in ['mean', 'sum', 'min', 'max']:
            raise ValueError("Unknown aggregation method: {agg}")
        self.agg = agg

        self.masking = masking

    def forward(self, loss: torch.Tensor, mask: Optional[torch.Tensor]):
        if self.masking and mask is not None:
            if mask.shape != loss.shape:
                raise ValueError(f"Loss shape and mask shape are different: {mask.shape} != {loss.shape}")
            
            return einops.reduce(loss[mask], '... ->', self.agg)   # one scalar
        else:
            return einops.reduce(loss, '... ->', self.agg)


class ObjectMaskCropping:
    """
        Crop object mask to disregard boundary regions of the object. Useful for excluding areas
        where finite difference calculations may be inaccurate due to discontinuities at object edges.
        
        Works by checking if all neighboring voxels within a specified padding distance are filled (i.e., part of the object).
        
        Parameters
        ----------
        padding : int, optional
            Number of voxels to crop from the object boundaries, by default 1.
    """
    
    def __init__(self, padding: int = 1):
        self.padding = padding
        self.padding_filter = torch.ones(
            [1, 1] + [self.padding*2 + 1]*3, dtype=torch.float32
        )

    def __call__(self, input_shape_mask: torch.Tensor) -> torch.Tensor:
        check_border = torch.nn.functional.conv3d(
            input_shape_mask.type(torch.float32),
            self.padding_filter,
            padding=self.padding,
        )
        return check_border == torch.sum(self.padding_filter)


class DiffFilterFactory:
    """
    Factory for generating finite difference filters for spatial derivatives.

    Creates convolutional filters that approximate spatial derivatives using finite difference
    methods. Supports computing gradients, divergence, and curl operators on multi-dimensional
    tensor fields.

    Parameters
    ----------
    accuracy : int, optional
        Order of accuracy for the finite difference approximation, by default 2.
    dx : float, optional
        Grid spacing for derivative calculation, by default 1.0.
    num_dims : int, optional
        Number of spatial dimensions, by default 3.
    dim_names : str, optional
        Names of the dimensions (e.g., 'xyz' for 3D), by default 'xyz'.

    Attributes
    ----------
    accuracy : int
        Order of accuracy for finite difference approximation.
    dx : float
        Grid spacing used in derivative computation.
    num_dims : int
        Number of spatial dimensions.
    dim_names : str
        Names assigned to each dimension.

    Raises
    ------
    ValueError
        If the length of dim_names does not match num_dims.
    """
    def __init__(self,
                 accuracy: int = 2,
                 dx: float = 1.0,
                 num_dims: int = 3,
                 dim_names: str = 'xyz'):
        if dx <= 0:
            raise ValueError(f"dx must be positive, got {dx}")
        if accuracy <= 0:
            raise ValueError(f"accuracy must be positive, got {accuracy}")
        if num_dims <= 0:
            raise ValueError(f"num_dims must be positive, got {num_dims}")

        self.accuracy = accuracy
        self.dx = dx
        self.num_dims = num_dims
        self.dim_names = dim_names

        if len(dim_names) != num_dims:
            raise ValueError(f"dim_names {dim_names} does not match num_dims {num_dims}")

    def _single_derivative_coeffs(self, order: int = 1) -> torch.Tensor:
        """
        Compute finite difference coefficients for a single derivative.

        Parameters
        ----------
        order : int, optional
            Order of the derivative, by default 1.

        Returns
        -------
        torch.Tensor
            Finite difference coefficients scaled by grid spacing.
        """
        if order == 0:
            return torch.tensor([1.0], dtype=torch.float32)
        coeffs = coefficients(deriv=order, acc=self.accuracy)
        return torch.tensor(
            coeffs['center']['coefficients'], dtype=torch.float32
        ) / (self.dx**order)

    def _generate_einops_expansion_expression(self, dim: int) -> str:
        """
        Generate einops pattern for expanding coefficients along a
        specific dimension.

        Parameters
        ----------
        dim : int
            Target dimension index for expansion.

        Returns
        -------
        str
            Einops rearrangement pattern string.
        """
        if dim >= self.num_dims:
            raise ValueError(f"dim {dim} must be less than num_dims {self.num_dims}")
        dims_before = ' '.join(['()']*dim)
        dims_after = ' '.join(['()']*(self.num_dims - dim - 1))
        return f'd -> {dims_before} d {dims_after}'

    def _pad_to_square(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pad tensor to make all dimensions equal to the maximum dimension size.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to pad.

        Returns
        -------
        torch.Tensor
            Padded tensor with all dimensions equal in size.
        """
        max_dim = max(tensor.shape)
        pad_sizes = [(max_dim - s) // 2 for s in tensor.shape]
        pad_sizes = [item for sublist in zip(pad_sizes, pad_sizes) for item in sublist]
        pad_sizes = pad_sizes[::-1]
        return torch.nn.functional.pad(tensor, pad_sizes, mode='constant', value=0)

    def derivative_from_expression(self, expression: str) -> torch.Tensor:
        """
        Compute the derivative coefficients from a given expression.

        Parameters
        ----------
        expression : str
            Expression representing the derivative using dimension names
            from dim_names. The count of each variable represents the
            order of the derivative with respect to that variable
            (e.g., 'xxy' represents ∂³/∂x²∂y).

        Returns
        -------
        torch.Tensor
            Finite difference filter coefficients as a
            multi-dimensional tensor.
        """
        orders = [expression.count(dim) for dim in self.dim_names]
        coeffs = [self._single_derivative_coeffs(order) for order in orders]
        coeffs = [
            einops.rearrange(
                coeff, self._generate_einops_expansion_expression(dim)
            )
            for dim, coeff in enumerate(coeffs)
        ]

        return reduce(mul, coeffs)

    def divergence(self) -> torch.Tensor:
        """
        Compute the divergence coefficients.

        Returns
        -------
        torch.Tensor
            A tensor containing the finite difference coefficients for
            computing the divergence of a num_dims-dimensional vector
            field.
        """
        per_dimension_coeffs = [
            self.derivative_from_expression(dim) for dim in self.dim_names
        ]
        per_dimension_coeffs = [
            self._pad_to_square(coeff) for coeff in per_dimension_coeffs
        ]
        divergence_filter = torch.stack(per_dimension_coeffs, dim=0)
        divergence_filter = einops.rearrange(divergence_filter, '... -> () ...')
        return divergence_filter

    def curl(self) -> torch.Tensor:
        """
        Compute the curl operator filter for 3D vector fields.

        Returns
        -------
        torch.Tensor
            Curl operator filter with shape (3, 3, k, k, k) where k
            depends on accuracy.
        """
        curl_filter = torch.zeros(
            (3, 3, self.accuracy + 1, self.accuracy + 1, self.accuracy + 1),
            dtype=torch.float32,
        )

        dy_filter = self.derivative_from_expression('y')
        dz_filter = self.derivative_from_expression('z')
        dy_padded = self._pad_to_square(dy_filter)
        dz_padded = self._pad_to_square(dz_filter)
        curl_filter[0, 2] = dy_padded
        curl_filter[0, 1] = -dz_padded

        dx_filter = self.derivative_from_expression('x')
        dx_padded = self._pad_to_square(dx_filter)
        curl_filter[1, 0] = dz_padded
        curl_filter[1, 2] = -dx_padded

        curl_filter[2, 1] = dx_padded
        curl_filter[2, 0] = -dy_padded

        return curl_filter


def mask_padding(input_shape_mask: torch.Tensor, padding: int = 1) -> torch.Tensor:
    """
    Convenience function for ObjectMaskPadding.

    Pads object mask to create a boundary region around objects.

    Parameters
    ----------
    input_shape_mask : torch.Tensor
        Input boolean mask tensor.
    padding : int, optional
        Padding distance in voxels, by default 1.

    Returns
    -------
    torch.Tensor
        Boolean tensor where True indicates all neighbors within
        padding distance are filled.
    """
    return ObjectMaskPadding(padding=padding)(input_shape_mask)
