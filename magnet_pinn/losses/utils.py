import torch
from typing import Optional, Union, Tuple, List
from findiff import coefficients
import einops
from operator import mul
from functools import reduce

class MaskedLossReducer(torch.nn.Module):
    def __init__(self):
        super(MaskedLossReducer, self).__init__()

    def forward(self, loss: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)
        else:
            if mask.shape != loss.shape:
                raise ValueError(f"mask shape {mask.shape} does not match loss shape {loss.shape}")
        return torch.mean(loss[mask])
    

class ObjectMaskPadding:
    def __init__(self, padding: int = 1):
        self.padding = padding
        self.padding_filter = torch.ones([1,1] + [self.padding*2 + 1]*3, dtype=torch.float32)

    def __call__(self, input_shape_mask: torch.Tensor) -> torch.Tensor:
        check_border = torch.nn.functional.conv3d(input_shape_mask.type(torch.float32), self.padding_filter, padding=self.padding)
        return check_border == torch.sum(self.padding_filter)

# TODO Add support for different dx values along different dimensions
class DiffFilterFactory:
    def __init__(self, 
                 accuracy: int = 2, 
                 dx: float = 1.0,
                 num_dims: int = 3,
                 dim_names: str = 'xyz'):
        self.accuracy = accuracy
        self.dx = dx
        self.num_dims = num_dims
        self.dim_names = dim_names

        if len(dim_names) != num_dims:
            raise ValueError(f"dim_names {dim_names} does not match num_dims {num_dims}")


    def _single_derivative_coeffs(self, order: int = 1) -> torch.Tensor:
        if order == 0:
            return torch.tensor([1.0], dtype=torch.float32)
        coeffs = coefficients(deriv=order, acc=self.accuracy)
        return torch.tensor(coeffs['center']['coefficients'], dtype=torch.float32)/(self.dx**order)
    
    def _generate_einops_expansion_expression(self, dim: int) -> str:
        if dim >= self.num_dims:
            raise ValueError(f"dim {dim} must be less than num_dims {self.num_dims}")
        dims_before = ' '.join(['()']*dim)
        dims_after = ' '.join(['()']*(self.num_dims - dim - 1))
        return f'd -> {dims_before} d {dims_after}'
    
    def _pad_to_square(self, tensor: torch.Tensor) -> torch.Tensor:
        max_dim = max(tensor.shape)
        pad_sizes = [(max_dim - s) // 2 for s in tensor.shape]
        pad_sizes = [item for sublist in zip(pad_sizes, pad_sizes) for item in sublist]
        pad_sizes = pad_sizes[::-1]  # reverse to match the order required by F.pad
        return torch.nn.functional.pad(tensor, pad_sizes, mode='constant', value=0)
        
    def derivative_from_expression(self, expression: str) -> torch.Tensor:
        """
        Compute the derivative coefficients from a given expression.

        Args:
            expression (str): The expression representing the derivative. The expression 
                should be a string containing the variable names as specified 
                in `self.dim_names`, where the count of each variable 
                represents the order of the derivative with respect to that 
                variable.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: A tuple containing the list of 
            coefficient tensors for each dimension and a tensor with a single element 1.0.
        """
        orders = [expression.count(dim) for dim in self.dim_names]
        coeffs = [self._single_derivative_coeffs(order) for order in orders]
        coeffs = [einops.rearrange(coeff, self._generate_einops_expansion_expression(dim)) for dim, coeff in enumerate(coeffs)]


        return reduce(mul, coeffs)
    
    def divergence(self) -> torch.Tensor:
        """
        Compute the divergence coefficients.

        Returns:
            torch.Tensor: A tensor containing the finite difference coefficients for 
            computing the divergence of a num_dims-dimensional vector field.
        """
        per_dimension_coeffs = [self.derivative_from_expression(dim) for dim in self.dim_names]
        per_dimension_coeffs = [self._pad_to_square(coeff) for coeff in per_dimension_coeffs]
        divergence_filter = torch.stack(per_dimension_coeffs, dim=0)
        divergence_filter = einops.rearrange(divergence_filter, '... -> () ...')
        return divergence_filter