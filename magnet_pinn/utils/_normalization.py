import torch
import numpy as np
import tqdm

from abc import ABC, abstractmethod


class Normalizer(ABC, torch.nn.Module):
    def __init__(self,
                 params: dict = {},
                 key: str = "input",
                 axis: int = 0,
                 ndims: int = 4):
        super().__init__()
        
        self._params = params
        self.counter = 0
        self._expand_params()
            
        self.key = key
        self.axis = axis
        self.ndims = ndims
    
    def forward(self, x):
        return self._normalize(x)
    
    def reverse(self,x):
        return self._denormalize(x)
    
    @abstractmethod
    def _normalize(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def _denormalize(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def _reset_params(self):
        raise NotImplementedError
    
    @abstractmethod
    def _update_params(self, x):
        raise NotImplementedError
    
    def fit_params(self, dataset, verbose: bool = True) -> None:
        self._reset_params()
        self.counter = 0
        iterator = tqdm.tqdm(dataset) if verbose else dataset

        for batch in iterator:
            x = batch[self.key]
            self._update_params(x)
            self.counter += 1
            
        self._expand_params()

    def load_from_numpy(self, path: str) -> None:
        self._params = np.load(path, allow_pickle=True).item()
        self._expand_params()
            
    
    @property
    def axes(self):
        return tuple(i for i in range(self.ndims) if i != self.axis)
    
    @property
    def params(self):
        return self._params
    
    def _expand_params(self):
        for key, value in self.params.items():
            self._params[key] = np.expand_dims(value, axis=self.axes)

    def get_params(self):
        params_dict = {}
        for key, value in self.params.items():
            params_dict[key] = np.squeeze(value, axis=self.axes)

        return params_dict
    
    
class MinMaxNormalizer(Normalizer):
    def _normalize(self, x):
        return (x - self.params['x_min']) / (self.params['x_max'] - self.params['x_min'])
    
    def _denormalize(self, x):
        return x * (self.params['x_max'] - self.params['x_min']) + self.params['x_min']
    
    def _reset_params(self):
        self.params["x_min"] = np.inf
        self.params["x_max"] = -np.inf

    def _update_params(self, x):
        self.params['x_min'] = np.minimum(self.params['x_min'], x.min(axis=self.axes))
        self.params['x_max'] = np.maximum(self.params['x_max'], x.max(axis=self.axes))
    

class StandardNormalizer(Normalizer):
    def _normalize(self, x):
        return (x - self.params['x_mean']) / self.params['x_var']**0.5
    
    def _denormalize(self, x):
        return x * self.params['x_var']**0.5 + self.params['x_mean']
    
    def _reset_params(self):
        self.params["x_mean"] = 0
        self.params["x_mean_sq"] = 0
        self.params["x_var"] = 1
        
    def _update_params(self, x):
        self.params["x_mean"] = self.counter / (self.counter + 1) * self.params["x_mean"] + x.mean(axis=self.axes) / (self.counter + 1)
        self.params["x_mean_sq"] = self.counter / (self.counter+ 1) * self.params["x_mean_sq"] + (x**2).mean(axis=self.axes) / (self.counter + 1)
        self.params["x_var"] = self.params["x_mean_sq"] - self.params["x_mean"]**2