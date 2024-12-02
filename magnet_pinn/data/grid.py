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
from einops import einsum, repeat

from typing import Tuple, Optional

import random
import torch

from .dataitem import DataItem
from .transforms import BaseTransform

from magnet_pinn.preprocessing.preprocessing import (
    VOXEL_SIZE_OUT_KEY,
    ANTENNA_MASKS_OUT_KEY,
    MIN_EXTENT_OUT_KEY,
    MAX_EXTENT_OUT_KEY,
    FEATURES_OUT_KEY,
    E_FIELD_OUT_KEY,
    H_FIELD_OUT_KEY,
    SUBJECT_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
    PROCESSED_ANTENNA_DIR_PATH,
    TRUNCATION_COEFFICIENTS_OUT_KEY,
    DTYPE_OUT_KEY
)




class MagnetGridIterator(torch.utils.data.IterableDataset):
    """
    Iterator for loading the magnetostatic simulation data.
    """
    def __init__(self, 
                 data_dir: str,
                 augmentation: Optional[BaseTransform] = None,
                 num_augmentations: int = 1):
        super().__init__()
        self.simulation_dir = os.path.join(data_dir, PROCESSED_SIMULATIONS_DIR_PATH)
        self.coils_path = os.path.join(data_dir, PROCESSED_ANTENNA_DIR_PATH, "antenna.h5")
        self.simulation_list = glob.glob(os.path.join(self.simulation_dir, "*.h5"))
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

        self.augmentation = augmentation
        self.num_augmentations = num_augmentations

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
    
    def _load_simulation(self, simulation_path: str) -> DataItem:
        """
        Loads simulation data from the h5 file.
        Parameters
        ----------
        index : int
            Index of the simulation file
        
        Returns
        -------
        DataItem
            DataItem object with the loaded data
        """
        with h5py.File(simulation_path) as f:
            field = self._read_fields(f, E_FIELD_OUT_KEY, H_FIELD_OUT_KEY)
            input_features = f[FEATURES_OUT_KEY][:]
            subject = f[SUBJECT_OUT_KEY][:]

            return DataItem(
                input=input_features,
                subject=np.max(subject, axis=-1),
                simulation=self._get_simulation_name(simulation_path),
                field=field,
                phase=np.zeros(self.num_coils),
                mask=np.ones(self.num_coils),
                coils=self.coils,
                dtype=f.attrs[DTYPE_OUT_KEY],
                truncation_coefficients=f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY]
            )
        

    def _read_fields(self, f: h5py.File, efield_key: str, hfield_key: str) -> npt.NDArray[np.float32]:
        def read_field(field_key: str) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            field_val = f[field_key][:]
            if field_val.dtype.names is None:
                return field_val.real, field_val.imag
            return field_val["re"], field_val["im"]
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
        re_efield, im_efield = read_field(efield_key)
        re_hfield, im_hfield = read_field(hfield_key)
        
        return np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)], axis=0)
    
    
    def __iter__(self):
        random.shuffle(self.simulation_list)
        for simulation in self.simulation_list:
            loaded_simulation = self._load_simulation(simulation)
            for i in range(self.num_augmentations):
                augmented_simulation = self.augmentation(loaded_simulation)
                yield augmented_simulation.__dict__
    
    def __len__(self):
        return len(self.simulation_list)*self.num_augmentations