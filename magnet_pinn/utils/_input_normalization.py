import torch

from abc import ABC, abstractmethod


class InputNormalizer(ABC, torch.nn.Module):
    def __init__(self,
                 params: dict = None):
        super().__init__()
        self.params = params
    
    def forward(self, x):
        return self._normalize(x)
    
    @abstractmethod
    def _normalize(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def _denormalize(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def _fit_params(self, dataset):
        raise NotImplementedError   
    
    
class InputMinMaxNormalizer(InputNormalizer):
    def _normalize(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)