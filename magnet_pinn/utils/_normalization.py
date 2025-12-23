"""Normalization utilities for tensor data preprocessing.

This module provides a flexible normalization framework for preprocessing tensor
data, particularly for machine learning applications in magnetostatic field
prediction. It includes:

- Abstract base classes for nonlinearities and normalizers
- Concrete nonlinearity implementations (Identity, Power, Log, Tanh, Arcsinh)
- Normalizer implementations (MinMax, Standard)
- MetaNormalizer for efficient multi-normalizer fitting

The normalization pipeline supports:

- Optional nonlinear transformations before/after normalization
- Online parameter estimation from iterable datasets
- JSON serialization for persistence
- Flexible axis-based normalization for multi-dimensional tensors
"""

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

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class Nonlinearity(ABC, torch.nn.Module):
    """Abstract base class for nonlinear transformations.

    Nonlinearities can be applied before or after normalization to improve
    the distribution of data. All implementations must provide forward and
    inverse transformations.
    """

    @abstractmethod
    def forward(self, x):
        """Apply the nonlinear transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x):
        """Apply the inverse nonlinear transformation.

        Parameters
        ----------
        x : torch.Tensor
            Transformed tensor.

        Returns
        -------
        torch.Tensor
            Original tensor.
        """
        raise NotImplementedError


class Identity(Nonlinearity):
    """Identity nonlinearity (no transformation).

    This is a no-op transformation that returns the input unchanged.
    Useful as a default when no nonlinearity is desired.
    """

    def forward(self, x):
        """Return input unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Same as input.
        """
        return x

    def inverse(self, x):
        """Return input unchanged (inverse of identity is identity).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Same as input.
        """
        return x


class Power(Nonlinearity):
    """Power nonlinearity transformation.

    Applies a power transformation while preserving the sign:
    f(x) = sign(x) * |x|^p

    This is useful for compressing or expanding the dynamic range of data.
    """

    def __init__(self, power: float = 2.0):
        """Initialize Power nonlinearity.

        Parameters
        ----------
        power : float, optional
            The exponent to apply. Must be positive. Default is 2.0.
        """
        super().__init__()
        self.power = power
        assert power > 0, "Power must be positive."

    def forward(self, x):
        """Apply power transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor with f(x) = sign(x) * |x|^power.
        """
        return torch.sign(x) * torch.abs(x) ** self.power

    def inverse(self, x):
        """Apply inverse power transformation.

        Parameters
        ----------
        x : torch.Tensor
            Transformed tensor.

        Returns
        -------
        torch.Tensor
            Original tensor with f^(-1)(x) = sign(x) * |x|^(1/power).
        """
        return torch.sign(x) * torch.abs(x) ** (1 / self.power)


class Log(Nonlinearity):
    """Logarithmic nonlinearity transformation.

    Applies a logarithmic transformation while preserving the sign:
    f(x) = sign(x) * log(1 + |x|)

    This is useful for compressing large values and is numerically stable
    near zero using log1p.
    """

    def forward(self, x):
        """Apply logarithmic transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor with f(x) = sign(x) * log(1 + |x|).
        """
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def inverse(self, x):
        """Apply inverse logarithmic transformation.

        Parameters
        ----------
        x : torch.Tensor
            Transformed tensor.

        Returns
        -------
        torch.Tensor
            Original tensor with f^(-1)(x) = sign(x) * (exp(|x|) - 1).
        """
        return torch.sign(x) * (torch.expm1(torch.abs(x)))


class Tanh(Nonlinearity):
    """Hyperbolic tangent nonlinearity.

    Applies tanh transformation: f(x) = tanh(x)

    This squashes values to the range (-1, 1) and is commonly used in
    neural networks. The inverse is the arctanh function.
    """

    def forward(self, x):
        """Apply hyperbolic tangent transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor with values in (-1, 1).
        """
        return torch.tanh(x)

    def inverse(self, x):
        """Apply inverse hyperbolic tangent (arctanh).

        Parameters
        ----------
        x : torch.Tensor
            Transformed tensor with values in (-1, 1).

        Returns
        -------
        torch.Tensor
            Original tensor with f^(-1)(x) = 0.5 * log((1+x)/(1-x)).
        """
        return 0.5 * torch.log((1 + x) / (1 - x))


class Arcsinh(Nonlinearity):
    """Inverse hyperbolic sine (arcsinh) nonlinearity.

    Applies arcsinh transformation: f(x) = asinh(x)

    This is similar to logarithmic transformation but smoother near zero.
    It's particularly useful for data with outliers or heavy tails.
    """

    def forward(self, x):
        """Apply inverse hyperbolic sine transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        return torch.asinh(x)

    def inverse(self, x):
        """Apply hyperbolic sine (inverse of arcsinh).

        Parameters
        ----------
        x : torch.Tensor
            Transformed tensor.

        Returns
        -------
        torch.Tensor
            Original tensor with f^(-1)(x) = sinh(x).
        """
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

    def __init__(
        self,
        params: dict = None,
        nonlinearity: Union[str,] = Identity(),
        nonlinearity_before: bool = False,
    ):
        """Initialize Normalizer.

        Parameters
        ----------
        params : dict, optional
            Pre-computed normalization parameters. If None, parameters must be
            fitted using fit_params().
        nonlinearity : Union[str, Nonlinearity], optional
            Nonlinearity to apply. Can be a Nonlinearity instance or a
            string name ('Identity', 'Power', 'Log', 'Tanh', 'Arcsinh').
            Default is Identity().
        nonlinearity_before : bool, optional
            If True, apply nonlinearity before normalization. If False, apply
            after normalization. Default is False.
        """
        super().__init__()

        self._params = params.copy() if params else {}
        self.nonlinearity = (
            nonlinearity
            if isinstance(nonlinearity, Nonlinearity)
            else self._get_nonlineartiy_function(nonlinearity)
        )
        self.nonlinearity_name = (
            nonlinearity
            if isinstance(nonlinearity, str)
            else nonlinearity.__class__.__name__
        )
        self.nonlinearity_before = nonlinearity_before
        self.counter = 0

    def forward(self, x, axis: int = 1):
        """Normalize the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize.
        axis : int, optional
            Channel axis along which normalization is performed. Default is 1.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        return self._normalize(x, axis=axis)

    def inverse(self, x, axis: int = 1):
        """Denormalize the input tensor (inverse transformation).

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor to denormalize.
        axis : int, optional
            Channel axis along which denormalization is performed. Default is 1.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        return self._denormalize(x, axis=axis)

    @abstractmethod
    def _normalize(self, x):
        """Internal method to perform normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def _denormalize(self, x):
        """Internal method to perform denormalization.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_params(self):
        """Reset normalization parameters to initial state.

        This method must be implemented by subclasses to reset their specific
        parameters before fitting on a new dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_params(self, x):
        """Update normalization parameters with a new batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data to update parameters with.

        Notes
        -----
        This method is called iteratively during fit_params() to accumulate
        statistics from the dataset.
        """
        raise NotImplementedError

    def fit_params(
        self,
        dataset: Iterable,
        axis: int = 0,
        key: str = "input",
        verbose: bool = True,
    ) -> None:
        """Fit normalization parameters from a dataset.

        Iterates through the dataset once to compute normalization parameters
        (e.g., min/max, mean/variance).

        Parameters
        ----------
        dataset : Iterable
            Iterable dataset where each item is a dictionary containing data.
        axis : int, optional
            Axis along which to compute statistics. Default is 0.
        key : str, optional
            Dictionary key to extract data from each batch. Default is 'input'.
        verbose : bool, optional
            If True, display a progress bar during fitting. Default is True.
        """
        self._reset_params()
        self.counter = 0
        iterator = tqdm.tqdm(dataset) if verbose else dataset

        for batch in iterator:
            x = batch[key]
            self._update_params(x, axis=axis)
            self.counter += 1

    def get_reduction_axes(self, ndims, axis):
        """Get axes for reduction operations excluding the specified axis.

        Parameters
        ----------
        ndims : int
            Number of dimensions in the tensor.
        axis : int
            Axis to exclude from reduction.

        Returns
        -------
        tuple of int
            Tuple of axis indices for reduction.
        """
        return tuple(i for i in range(ndims) if i != axis)

    @property
    def params(self):
        """Get normalization parameters.

        Returns
        -------
        dict
            Dictionary of normalization parameters.
        """
        return self._params

    def _expand_params(self, params_dict: dict = None, axis: int = 0, ndims: int = 5):
        """Expand parameter tensors to match input dimensionality.

        Adds singleton dimensions to parameter tensors so they can be
        broadcast with multi-dimensional inputs during normalization.

        Parameters
        ----------
        params_dict : dict, optional
            Dictionary of parameters to expand. If None, uses self._params.
        axis : int, optional
            Channel axis position. Default is 0.
        ndims : int, optional
            Target number of dimensions. Default is 5.

        Returns
        -------
        dict
            Dictionary with expanded parameter tensors.
        """
        if params_dict is None:
            params_dict = self._params

        expanded_params = {}
        for key, value in params_dict.items():
            pattern = (
                "c -> "
                + " ".join(["1"] * axis)
                + " c "
                + " ".join(["1"] * (ndims - axis - 1))
            )
            expanded_params[key] = einops.rearrange(value, pattern)

        return expanded_params

    def _cast_params(
        self,
        params_dict: dict = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """Cast parameter values to specified dtype and device.

        Converts stored parameters (typically Python lists/numpy arrays) to
        PyTorch tensors with the appropriate dtype and device for
        computation.

        Parameters
        ----------
        params_dict : dict, optional
            Dictionary of parameters to cast. If None, uses self._params.
        dtype : torch.dtype, optional
            Target data type. Default is torch.float32.
        device : torch.device, optional
            Target device. Default is CPU.

        Returns
        -------
        dict
            Dictionary with casted parameter tensors.
        """
        if params_dict is None:
            params_dict = self._params

        casted_params = {}
        for key, value in params_dict.items():
            casted_params[key] = torch.tensor(value, dtype=dtype, device=device)

        return casted_params

    def save_as_json(self, path: str):
        """Save normalization parameters to a JSON file.

        Parameters
        ----------
        path : str
            File path to save the parameters. Parent directories are created
            if they don't exist.

        Notes
        -----
        The saved JSON includes normalization parameters plus metadata about
        the nonlinearity function and its application order.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            params = self._params.copy()
            # add nonlinearity info
            params["nonlinearity"] = self.nonlinearity_name
            params["nonlinearity_before"] = self.nonlinearity_before
            json.dump(params, f)

    def _get_nonlineartiy_function(name: str = "Identity"):
        """Get nonlinearity instance from string name.

        Parameters
        ----------
        name : str, optional
            Name of the nonlinearity ('Identity', 'Power', 'Log', 'Tanh',
            'Arcsinh'). Default is 'Identity'.

        Returns
        -------
        Nonlinearity
            Instance of the requested nonlinearity.

        Raises
        ------
        ValueError
            If the nonlinearity name is unknown.
        """
        if name == "Identity":
            return Identity()
        elif name == "Power":
            return Power()
        elif name == "Log":
            return Log()
        elif name == "Tanh":
            return Tanh()
        elif name == "Arcsinh":
            return Arcsinh()
        else:
            raise ValueError(f"Unknown nonlinearity: {name}")

    @classmethod
    def load_from_json(cls, path: str) -> Self:
        """Load normalization parameters from a JSON file.

        Parameters
        ----------
        path : str
            File path to load parameters from.

        Returns
        -------
        Self
            Instance of the normalizer class with loaded parameters.

        Notes
        -----
        The JSON file must have been created by save_as_json() and contain
        both normalization parameters and nonlinearity metadata.
        """
        with open(path, "r") as f:
            params = json.load(f)
            nonlinearity = params.pop("nonlinearity", "Identity")
            nonlinearity_before = params.pop("nonlinearity_before", False)
            nonlinearity_fn = cls._get_nonlineartiy_function(nonlinearity)

        return cast(
            Self,
            cls(
                params=params,
                nonlinearity=nonlinearity_fn,
                nonlinearity_before=nonlinearity_before,
            ),
        )


class MinMaxNormalizer(Normalizer):
    """
    Min-Max Normalizer

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the normalizer
    """

    def _normalize(self, x, axis: int = 0):
        """Normalize using min-max scaling.

        Scales values to [0, 1] range: (x - min) / (max - min)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize.
        axis : int, optional
            Channel axis. Default is 0.

        Returns
        -------
        torch.Tensor
            Normalized tensor with values in [0, 1].
        """
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        if self.nonlinearity_before:
            x_nl = self.nonlinearity(x)
            return (x_nl - params["x_min"]) / (params["x_max"] - params["x_min"])
        else:
            x_norm = (x - params["x_min"]) / (params["x_max"] - params["x_min"])
            return self.nonlinearity(x_norm)

    def _denormalize(self, x, axis: int = 0):
        """Denormalize from [0, 1] range back to original scale.

        Inverse of min-max scaling: x * (max - min) + min

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor with values in [0, 1].
        axis : int, optional
            Channel axis. Default is 0.

        Returns
        -------
        torch.Tensor
            Denormalized tensor in original scale.
        """
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        if self.nonlinearity_before:
            x_denorm = x * (params["x_max"] - params["x_min"]) + params["x_min"]
            return self.nonlinearity.inverse(x_denorm)
        else:
            x_nl = self.nonlinearity.inverse(x)
            return x_nl * (params["x_max"] - params["x_min"]) + params["x_min"]

    def _reset_params(self):
        """Reset min and max parameters to extreme values.

        Sets min to +inf and max to -inf so first batch will set the initial
        values correctly.
        """
        self._params["x_min"] = [float("inf")]
        self._params["x_max"] = [float("-inf")]

    def _update_params(self, x, axis: int = 0):
        """Update min and max values from a new batch.

        Tracks global minimum and maximum across all batches seen so far.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data.
        axis : int, optional
            Channel axis to compute statistics along. Default is 0.
        """
        pattern = " ".join(ALPHABET[:axis]) + " c ... -> c"
        if self.nonlinearity_before:
            x = self.nonlinearity(x)
        cur_min = einops.reduce(x, pattern, reduction="min").tolist()
        cur_max = einops.reduce(x, pattern, reduction="max").tolist()
        self._params["x_min"] = [
            min(prev, cur)
            for prev, cur in zip_longest(
                self._params["x_min"], cur_min, fillvalue=float("inf")
            )
        ]
        self._params["x_max"] = [
            max(prev, cur)
            for prev, cur in zip_longest(
                self._params["x_max"], cur_max, fillvalue=float("-inf")
            )
        ]


class StandardNormalizer(Normalizer):
    """
    Standard Normalizer

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the normalizer
    """

    def _normalize(self, x, axis: int = 0):
        """Normalize using z-score standardization.

        Transforms to zero mean and unit variance: (x - mean) / std

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize.
        axis : int, optional
            Channel axis. Default is 0.

        Returns
        -------
        torch.Tensor
            Normalized tensor with ~zero mean and ~unit variance.
        """
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        params["x_var"] = params["x_mean_sq"] - params["x_mean"] ** 2
        if self.nonlinearity_before:
            x_nl = self.nonlinearity(x)
            return (x_nl - params["x_mean"]) / params["x_var"] ** 0.5
        else:
            x_norm = (x - params["x_mean"]) / params["x_var"] ** 0.5
            return self.nonlinearity(x_norm)

    def _denormalize(self, x, axis: int = 0):
        """Denormalize from z-score back to original scale.

        Inverse transformation: x * std + mean

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor with ~zero mean and ~unit variance.
        axis : int, optional
            Channel axis. Default is 0.

        Returns
        -------
        torch.Tensor
            Denormalized tensor in original scale.
        """
        params = self._cast_params(dtype=x.dtype, device=x.device)
        params = self._expand_params(params, axis=axis, ndims=x.ndim)
        params["x_var"] = params["x_mean_sq"] - params["x_mean"] ** 2
        if self.nonlinearity_before:
            x_denorm = x * params["x_var"] ** 0.5 + params["x_mean"]
            return self.nonlinearity.inverse(x_denorm)
        else:
            x_nl = self.nonlinearity.inverse(x)
            return x_nl * params["x_var"] ** 0.5 + params["x_mean"]

    def _reset_params(self):
        """Reset mean and variance tracking parameters to zero.

        Initializes online mean and mean-square estimators.
        """
        self._params["x_mean"] = [0]
        self._params["x_mean_sq"] = [0]

    def _update_params(self, x, axis: int = 0):
        """Update mean and variance estimates with online algorithm.

        Uses Welford's online algorithm to maintain running mean and
        mean-of-squares, from which variance is computed.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data.
        axis : int, optional
            Channel axis to compute statistics along. Default is 0.
        """

        def mean_update(prev_avg, cur_avg, counter):
            """Compute incremental mean update.

            Parameters
            ----------
            prev_avg : float
                Previous running average.
            cur_avg : float
                Current batch average.
            counter : int
                Number of previous batches seen.

            Returns
            -------
            float
                Updated running average.
            """
            return counter / (counter + 1) * prev_avg + cur_avg / (counter + 1)

        if self.nonlinearity_before:
            x = self.nonlinearity(x)
        pattern = " ".join(ALPHABET[:axis]) + " c ... -> c"
        cur_mean = einops.reduce(x, pattern, reduction="mean").tolist()
        cur_mean_sq = einops.reduce(x**2, pattern, reduction="mean").tolist()
        self._params["x_mean"] = [
            mean_update(prev, cur, self.counter)
            for prev, cur in zip_longest(self._params["x_mean"], cur_mean, fillvalue=0)
        ]
        self._params["x_mean_sq"] = [
            mean_update(prev, cur, self.counter)
            for prev, cur in zip_longest(
                self._params["x_mean_sq"], cur_mean_sq, fillvalue=0
            )
        ]


class MetaNormalizer(Normalizer):
    """
    MetaNormalizer to fit multiple normalizers in one loop over the dataset.

    Purpose
    -------
    The MetaNormalizer is designed to streamline the process of fitting
    multiple normalizers (e.g., MinMaxNormalizer, StandardNormalizer)
    simultaneously in a single pass over the dataset. This is
    particularly useful when iterating over the dataset is
    time-consuming, as it avoids the need for multiple iterations.

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
        """Initialize MetaNormalizer with a list of normalizers.

        Parameters
        ----------
        normalizers : list
            List of Normalizer instances to manage and fit simultaneously.
        """
        self.normalizers = normalizers
        self.counter = 0  # MetaNormalizer's counter

    def _normalize(self, x, axis: int = 0):
        """Not implemented for MetaNormalizer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        axis : int, optional
            Channel axis. Default is 0.

        Raises
        ------
        NotImplementedError
            MetaNormalizer does not support direct normalization.
        """
        raise NotImplementedError(
            "MetaNormalizer does not support direct normalization."
        )

    def _denormalize(self, x, axis: int = 0):
        """Not implemented for MetaNormalizer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        axis : int, optional
            Channel axis. Default is 0.

        Raises
        ------
        NotImplementedError
            MetaNormalizer does not support direct denormalization.
        """
        raise NotImplementedError(
            "MetaNormalizer does not support direct denormalization."
        )

    def _expand_params(self, params_dict: dict = None, axis: int = 0, ndims: int = 5):
        """Not implemented for MetaNormalizer.

        Parameters
        ----------
        params_dict : dict, optional
            Parameter dictionary.
        axis : int, optional
            Channel axis. Default is 0.
        ndims : int, optional
            Number of dimensions. Default is 5.

        Raises
        ------
        NotImplementedError
            MetaNormalizer does not support parameter expansion.
        """
        raise NotImplementedError(
            "MetaNormalizer does not support parameter expansion."
        )

    def _cast_params(
        self,
        params_dict: dict = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """Not implemented for MetaNormalizer.

        Parameters
        ----------
        params_dict : dict, optional
            Parameter dictionary.
        dtype : torch.dtype, optional
            Target data type. Default is torch.float32.
        device : torch.device, optional
            Target device. Default is CPU.

        Raises
        ------
        NotImplementedError
            MetaNormalizer does not support parameter casting.
        """
        raise NotImplementedError("MetaNormalizer does not support parameter casting.")

    def _reset_params(self):
        """Reset parameters for all managed normalizers."""
        for normalizer in self.normalizers:
            normalizer._reset_params()

    def fit_params(
        self,
        dataset: Iterable,
        axis: int = 0,
        keys: Union[str, List[str]] = "input",
        verbose: bool = True,
    ) -> None:
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
                raise ValueError(
                    "The number of keys must match the number of normalizers."
                )
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

    def save_as_json(
        self, file_names: List[str], base_path: Optional[str] = None
    ) -> None:
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
            raise ValueError(
                "The number of file names must match the number of normalizers."
            )

        if base_path is not None and not isinstance(base_path, str):
            raise TypeError("base_path must be a string or None.")

        for i, (normalizer, file_name) in enumerate(zip(self.normalizers, file_names)):
            if base_path:
                file_name = os.path.join(base_path, file_name)
            normalizer.save_as_json(file_name)
