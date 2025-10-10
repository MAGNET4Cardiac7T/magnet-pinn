import torch
import tqdm
import einops
import numpy as np

from abc import ABC, abstractmethod
from typing import Iterable, cast, Union, List, Optional
from typing_extensions import Self
from itertools import zip_longest

import json
import os

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

class Nonlinearity(ABC,torch.nn.Module):
    @abstractmethod
    def forward(self, x):
        """ """
        raise NotImplementedError
    
    @abstractmethod
    def inverse(self, x):
        raise NotImplementedError
    
class Identity(Nonlinearity):
    """
    Identity nonlinearity
    """
    def forward(self, x):
        """ """
        return x
    
    def inverse(self, x):
        return x
    
class Power(Nonlinearity):
    """
    Power nonlinearity
    """
    def __init__(self, power: float = 2.0):
        super().__init__()
        self.power = power
        assert power > 0, "Power must be positive."
    def forward(self, x):
        """ """
        return torch.sign(x) * torch.abs(x)**self.power
    def inverse(self, x):
        return torch.sign(x) * torch.abs(x)**(1/self.power)
    
class Log(Nonlinearity):
    """
    Logarithmic nonlinearity
    """
    def forward(self, x):
        """ """
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    def inverse(self, x):
        return torch.sign(x) * (torch.expm1(torch.abs(x)))
    
class Tanh(Nonlinearity):
    """
    Hyperbolic tangent nonlinearity
    """
    def forward(self, x):
        """ """
        return torch.tanh(x)
    
    def inverse(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

class Arcsinh(Nonlinearity):
    """
    Inverse hyperbolic sine nonlinearity
    """
    def forward(self, x):
        """ """
        return torch.asinh(x)
    
    def inverse(self, x):
        return torch.sinh(x)

class Normalizer(torch.nn.Module):
    """
    Base class for normalizers
    
    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the normalizer
    nonlinearity : Union[str, Nonlinearity]
        Nonlinearity to be applied before/after normalization
    nonlinearity_before : bool
        If True, apply nonlinearity before normalization, else after
    """
    def __init__(self,
                 params: dict = None,
                 nonlinearity: Union[str, ] = Identity(),
                 nonlinearity_before: bool = False,
                 ):
        super().__init__()
        
        self._params = params.copy() if params else {}
        self.nonlinearity = nonlinearity if isinstance(nonlinearity, Nonlinearity) else self._get_nonlineartiy_function(nonlinearity)
        self.nonlinearity_name = nonlinearity if isinstance(nonlinearity, str) else nonlinearity.__class__.__name__
        self.nonlinearity_before = nonlinearity_before
        self.counter = 0
    
    def forward(self, x, axis: int = 1):
        """ """
        return self._normalize(x, axis=axis)
    
    def inverse(self,x, axis: int = 1):
        return self._denormalize(x, axis=axis)
    
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
    
    def fit_params(self, 
                   dataset: Iterable,
                   axis: int = 0,
                   key: str = 'input',
                   verbose: bool = True,
                   ) -> None:
        self._reset_params()
        self.counter = 0
        iterator = tqdm.tqdm(dataset) if verbose else dataset

        for batch in iterator:
            x = batch[key]
            self._update_params(x, axis=axis)
            self.counter += 1
    
    def get_reduction_axes(self, ndims, axis):
        return tuple(i for i in range(ndims) if i != axis)
    
    @property
    def params(self):
        return self._params
    
    def _expand_params(self, params_dict: dict = None, axis: int = 0, ndims: int = 5):
        if params_dict is None:
            params_dict = self._params

        expanded_params = {}
        for key, value in params_dict.items():
            pattern = 'c -> ' + ' '.join(['1'] * axis) + ' c ' + ' '.join(['1'] * (ndims - axis - 1))
            expanded_params[key] = einops.rearrange(value, pattern)
            
        return expanded_params
    
    def _cast_params(self, params_dict: dict = None, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):
        if params_dict is None:
            params_dict = self._params

        casted_params = {}
        for key, value in params_dict.items():
            casted_params[key] = torch.tensor(value, dtype=dtype, device=device)
        
        return casted_params
    
    def save_as_json(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            params = self._params.copy()
            # add nonlinearity info
            params['nonlinearity'] = self.nonlinearity_name
            params['nonlinearity_before'] = self.nonlinearity_before
            json.dump(params, f)

    def _get_nonlineartiy_function(name: str = "Identity"):
        if name == 'Identity':
            return Identity()
        elif name == 'Power':
            return Power()
        elif name == 'Log':
            return Log()
        elif name == 'Tanh':
            return Tanh()
        elif name == 'Arcsinh':
            return Arcsinh()
        else:
            raise ValueError(f"Unknown nonlinearity: {name}")

    @classmethod
    def load_from_json(cls, path: str) -> Self:
        with open(path, 'r') as f:
            params = json.load(f)    
            nonlinearity = params.pop('nonlinearity', 'Identity')
            nonlinearity_before = params.pop('nonlinearity_before', False)
            nonlinearity_fn = cls._get_nonlineartiy_function(nonlinearity)

        return cast(Self, cls(params=params, nonlinearity=nonlinearity_fn, nonlinearity_before=nonlinearity_before))

class MinMaxNormalizer(Normalizer):
    """
    Min-Max Normalizer

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the normalizer
    """
    def _normalize(self, x, axis: int = 0):
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        if self.nonlinearity_before:
            x_nl = self.nonlinearity(x)
            return (x_nl - params['x_min']) / (params['x_max'] - params['x_min'])
        else:
            x_norm = (x - params['x_min']) / (params['x_max'] - params['x_min'])
            return self.nonlinearity(x_norm)
    
    def _denormalize(self, x, axis: int = 0):
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        if self.nonlinearity_before:
            x_denorm =  x * (params['x_max'] - params['x_min']) + params['x_min']
            return self.nonlinearity.inverse(x_denorm)
        else:
            x_nl = self.nonlinearity.inverse(x)
            return x_nl * (params['x_max'] - params['x_min']) + params['x_min']
    
    def _reset_params(self):
        self._params["x_min"] = [float('inf')]
        self._params["x_max"] = [float('-inf')]

    def _update_params(self, x, axis: int = 0):
        pattern = ' '.join(ALPHABET[:axis]) + ' c ... -> c'
        if self.nonlinearity_before:
            x = self.nonlinearity(x)
        cur_min = einops.reduce(x, pattern, reduction='min').tolist()
        cur_max = einops.reduce(x, pattern, reduction='max').tolist()
        self._params['x_min'] = [min(prev, cur) for prev, cur in zip_longest(self._params['x_min'], cur_min, fillvalue=float('inf'))]
        self._params['x_max'] = [max(prev, cur) for prev, cur in zip_longest(self._params['x_max'], cur_max, fillvalue=float('-inf'))]
    
class StandardNormalizer(Normalizer):
    """
    Standard Normalizer

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the normalizer
    """
    def _normalize(self, x, axis: int = 0):
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        params["x_var"] = params["x_mean_sq"] - params["x_mean"]**2
        return (x - params['x_mean']) / params['x_var']**0.5
    
    def _denormalize(self, x, axis: int = 0):
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        params["x_var"] = params["x_mean_sq"] - params["x_mean"]**2
        return x * params['x_var']**0.5 + params['x_mean']
    
    def _reset_params(self):
        self._params["x_mean"] = [0]
        self._params["x_mean_sq"] = [0]

    def _update_params(self, x, axis: int = 0):
        def mean_update(prev_avg, cur_avg, counter):
            return counter / (counter + 1) * prev_avg + cur_avg / (counter + 1)
        if self.nonlinearity_before:
            x = self.nonlinearity(x)
        pattern = ' '.join(ALPHABET[:axis]) + ' c ... -> c'
        cur_mean = einops.reduce(x, pattern, reduction='mean').tolist()
        cur_mean_sq = einops.reduce(x**2, pattern, reduction='mean').tolist()
        self._params["x_mean"] = [mean_update(prev, cur, self.counter) for prev, cur in zip_longest(self._params["x_mean"], cur_mean, fillvalue=0)]
        self._params["x_mean_sq"] = [mean_update(prev, cur, self.counter) for prev, cur in zip_longest(self._params["x_mean_sq"], cur_mean_sq, fillvalue=0)]

class MetaNormalizer(Normalizer):
    """
    MetaNormalizer to fit multiple normalizers in one loop over the dataset.

    Purpose
    -------
    The MetaNormalizer is designed to streamline the process of fitting multiple
    normalizers (e.g., MinMaxNormalizer, StandardNormalizer) simultaneously in a
    single pass over the dataset. This is particularly useful when iterating over
    the dataset is time-consuming, as it avoids the need for multiple iterations.

    Functionality
    -------------
    - The MetaNormalizer manages a list of normalizers.
    - It ensures that each normalizer is updated with the appropriate data during
      the fitting process.
    - It supports using the same or different keys for extracting data for each
      normalizer.

    Parameters
    ----------
    normalizers : list
        List of normalizer instances to be managed by MetaNormalizer.

    Methods
    -------
    fit_params(dataset, axis=0, keys="input", verbose=True):
        Fits the parameters of all normalizers in one loop over the dataset.
    save_as_json(base_path):
        Saves all normalizers separately to the specified base path.
    """
    def __init__(self, normalizers: list):
        self.normalizers = normalizers
        self.counter = 0  # MetaNormalizer's counter

    def _normalize(self, x, axis: int = 0):
        raise NotImplementedError("MetaNormalizer does not support direct normalization.")

    def _denormalize(self, x, axis: int = 0):
        raise NotImplementedError("MetaNormalizer does not support direct denormalization.")

    def _expand_params(self, params_dict: dict = None, axis: int = 0, ndims: int = 5):
        raise NotImplementedError("MetaNormalizer does not support parameter expansion.")

    def _cast_params(self, params_dict: dict = None, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):
        raise NotImplementedError("MetaNormalizer does not support parameter casting.")

    def _reset_params(self):
        for normalizer in self.normalizers:
            normalizer._reset_params()

    def fit_params(self, 
                   dataset: Iterable, 
                   axis: int = 0, 
                   keys: Union[str, List[str]] = "input", 
                   verbose: bool = True) -> None:
        """
        Fit parameters for all normalizers in one loop over the dataset.

        Parameters
        ----------
        dataset : Iterable
            Dataset to fit the normalizers on.
        axis : int
            Axis along which to normalize.
        keys : Union[str, List[str]]
            Key(s) to extract data for each normalizer. If a single string is provided,
            it is used for all normalizers. If a list is provided, it must have the same
            length as the number of normalizers.
        verbose : bool
            Whether to display a progress bar.
        """
        self.counter = 0  # Reset MetaNormalizer's counter
        if isinstance(keys, str):
            keys = [keys] * len(self.normalizers)
        elif isinstance(keys, list):
            if len(keys) != len(self.normalizers):
                raise ValueError("The number of keys must match the number of normalizers.")
        else:
            raise TypeError("Keys must be either a string or a list of strings.")

        self._reset_params()
        iterator = tqdm.tqdm(dataset) if verbose else dataset

        for batch in iterator:
            for normalizer, key in zip(self.normalizers, keys):
                x = batch[key]
                normalizer.counter = self.counter
                normalizer._update_params(x, axis=axis)
            self.counter += 1  # Increment MetaNormalizer's counter

    def save_as_json(self, file_names: List[str], base_path: Optional[str] = None) -> None:
        """
        Save all normalizers separately to the specified file names.

        Parameters
        ----------
        file_names : List[str]
            List of file names for saving each normalizer. The length of the list
            must match the number of normalizers.
        base_path : Optional[str], optional
            The base directory to prepend to each file name. If None, only the
            file names are used. If provided, it must be a string.
        """
        if len(file_names) != len(self.normalizers):
            raise ValueError("The number of file names must match the number of normalizers.")

        if base_path is not None and not isinstance(base_path, str):
            raise TypeError("base_path must be a string or None.")

        for i, (normalizer, file_name) in enumerate(zip(self.normalizers, file_names)):
            if base_path:
                file_name = os.path.join(base_path, file_name)
            normalizer.save_as_json(file_name)