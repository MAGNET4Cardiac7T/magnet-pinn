import torch
from typing import Optional, Union, Tuple, List


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
    
# TODO Calculate findiffs at different accuracies
def partial_derivative_findiff(accuracy: int = 2) -> List[float]:
    if accuracy == 2:
        return [-0.5, 0, 0.5]
    else:
        raise ValueError(f"accuracy {accuracy} not supported")