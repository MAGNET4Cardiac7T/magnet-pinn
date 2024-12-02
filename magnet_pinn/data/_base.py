"""
NAME
    dataset.py
DESCRIPTION
    This module contains classes for loading the magnetostatic simulation data.
"""
import os
import h5py


import glob
import numpy as np
import numpy.typing as npt

from typing import Tuple, Optional
from abc import ABC, abstractmethod

import random
import torch

from .dataitem import DataItem
from .transforms import BaseTransform

from magnet_pinn.preprocessing.preprocessing import (
    ANTENNA_MASKS_OUT_KEY,
    FEATURES_OUT_KEY,
    E_FIELD_OUT_KEY,
    H_FIELD_OUT_KEY,
    SUBJECT_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
    PROCESSED_ANTENNA_DIR_PATH,
    TRUNCATION_COEFFICIENTS_OUT_KEY,
    DTYPE_OUT_KEY
)


class MagnetBaseIterator(torch.utils.data.IterableDataset, ABC):
    """
    Iterator for loading the magnetostatic simulation data.
    """
    def __init__(self, 
                 data_dir: str,
                 transforms: Optional[BaseTransform] = None,
                 num_samples: int = 1):
        super().__init__()
        self.simulation_dir = os.path.join(data_dir, PROCESSED_SIMULATIONS_DIR_PATH)
        self.coils_path = os.path.join(data_dir, PROCESSED_ANTENNA_DIR_PATH, "antenna.h5")
        self.simulation_list = glob.glob(os.path.join(self.simulation_dir, "*.h5"))
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

        self.transforms = transforms
        self.num_samples = num_samples

    def _get_simulation_name(self, simulation) -> str:
        return os.path.basename(simulation)[:-3]

    def _read_coils(self) -> npt.NDArray[np.bool_]:
        """
        Method reads coils masks from the h5 file.

        Returns
        -------
        npt.NDArray[np.bool_]
            Coils masks array
        """
        with h5py.File(self.coils_path) as f:
            coils = f[ANTENNA_MASKS_OUT_KEY][:]
        return coils
    
    @abstractmethod
    def _load_simulation(self, simulation_path: str) -> DataItem:
        raise NotImplementedError("This method should be implemented in the derived class")
        

    def _read_fields(self, simulation_path: str) -> npt.NDArray[np.float32]:
        """
        A method for reading the field from the h5 file.
        Reads and splits the field into real and imaginary parts.

        Parameters
        ----------
        f : h5py.File
            h5 file desc    pass

        Returns
        -------
        Dict
            A dictionary with `re_field_key` and `im_field_key` keys
            with real and imaginary parts of the field
        """

        def read_field(f: h5py.File, field_key: str) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            field_val = f[field_key][:]
            if field_val.dtype.names is None:
                return field_val.real, field_val.imag
            return field_val["re"], field_val["im"]
        
        with h5py.File(simulation_path) as f:
            re_efield, im_efield = read_field(f, E_FIELD_OUT_KEY)
            re_hfield, im_hfield = read_field(f, H_FIELD_OUT_KEY)
        
        return np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)], axis=0)
    
    def _read_input(self, simulation_path: str) -> npt.NDArray[np.float32]:
        """
        Method reads input features from the h5 file.

        Returns
        -------
        npt.NDArray[np.float32]
            Input features array
        """
        with h5py.File(simulation_path) as f:
            features = f[FEATURES_OUT_KEY][:]
        return features
    
    def _read_subject(self, simulation_path: str) -> npt.NDArray[np.bool_]:
        """
        Method reads the subject mask from the h5 file.

        Returns
        -------
        npt.NDArray[np.bool_]
            Subject array
        """
        with h5py.File(simulation_path) as f:
            subject = f[SUBJECT_OUT_KEY][:]
        subject = np.max(subject, axis=-1)
        return subject
    
    def _get_dtype(self, simulation_path: str) -> str:
        """
        Method reads the dtype from the h5 file.

        Returns
        -------
        str
            dtype
        """
        with h5py.File(simulation_path) as f:
            dtype = f.attrs[DTYPE_OUT_KEY]
        return dtype
    
    def _get_truncation_coefficients(self, simulation_path: str) -> npt.NDArray:
        """
        Method reads the truncation coefficients from the h5 file.

        Returns
        -------
        npt.NDArray
            Truncation coefficients
        """
        with h5py.File(simulation_path) as f:
            truncation_coefficients = f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY]
        return truncation_coefficients
    
    def __iter__(self):
        random.shuffle(self.simulation_list)
        for simulation in self.simulation_list:
            loaded_simulation = self._load_simulation(simulation)
            for i in range(self.num_samples):
                augmented_simulation = self.transforms(loaded_simulation)
                yield augmented_simulation.__dict__
    
    def __len__(self):
        return len(self.simulation_list)*self.num_samples