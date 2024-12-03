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