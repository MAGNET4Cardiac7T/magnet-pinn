import torch
import numpy as np
import tqdm

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
    def fit_params(self, dataset):
        raise NotImplementedError   
    
    
class InputMinMaxNormalizer(InputNormalizer):
    def _normalize(self, x):
        return (x - self.params['x_min']) / (self.params['x_max'] - self.params['x_min'])
    
    def _denormalize(self, x):
        return x * (self.params['x_max'] - self.params['x_min']) + self.params['x_min']
    
    def fit_params(self, dataset):
        x_min = np.inf
        x_max = np.NINF

        for batch in tqdm.tqdm(dataset):
            x = batch['input']
            print(x.shape)
            x_min = np.minimum(x_min, x.min(axis=(0, 2, 3)))
            x_max = np.maximum(x_max, x.max(axis=(0, 2, 3)))

        
        self.params = {'x_min': x_min, 'x_max': x_max}
        return self.params
    

class InputStandardNormalizer(InputNormalizer):
    def _normalize(self, x):
        return (x - self.params['x_mean']) / self.params['x_std']
    
    def _denormalize(self, x):
        return x * self.params['x_std'] + self.params['x_mean']
    
    def fit_params(self, dataset):
        x_mean = 0
        x_std = 0
        n = 0

        for batch in tqdm.tqdm(dataset):
            x = batch['input']
            n += x.shape[0]
            x_mean += x.sum(axis=(0, 2, 3))
            x_std += (x**2).sum(axis=(0, 2, 3))
        
        x_mean /= n
        x_std = np.sqrt(x_std / n - x_mean**2)
        
        self.params = {'x_mean': x_mean, 'x_std': x_std}
        return self.params