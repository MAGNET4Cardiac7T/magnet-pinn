import torch
from typing import Optional, Union, Tuple, List
from findiff import coefficients


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
    
def mask_padding(input_shape_mask: torch.Tensor, padding: int = 1) -> torch.Tensor:
    padding_filter = torch.ones([1,1] + [padding*2 + 1]*3, dtype=torch.float32)
    check_border = torch.nn.functional.conv3d(input_shape_mask.type(torch.float32), padding_filter, padding=padding)
    return check_border == torch.sum(padding_filter)

class DiffFilterFactory:
    def __init__(self, accuracy: int = 2):
        self.accuracy = accuracy

    def _single_derivative_coeffs(self, order: int = 1) -> torch.Tensor:
        if order == 0:
            return torch.tensor([1.0], dtype=torch.float32)
        coeffs = coefficients(deriv=order, acc=self.accuracy)
        return torch.tensor(coeffs['center']['coefficients'], dtype=torch.float32)
    

if __name__ == '__main__':
    
    diff_filter_factory = DiffFilterFactory()
    print(diff_filter_factory._single_derivative_coeffs(1))