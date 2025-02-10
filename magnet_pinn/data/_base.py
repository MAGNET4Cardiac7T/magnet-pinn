"""
NAME
    dataset.py
DESCRIPTION
    This module contains classes for loading the magnetostatic simulation data.
"""
import os
import h5py
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from natsort import natsorted

from typing import Tuple, Optional
from abc import ABC, abstractmethod

import random
import torch

from .dataitem import DataItem
from .transforms import BaseTransform, DefaultTransform, check_transforms

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
                 data_dir: Union[str, Path],
                 transforms: Optional[BaseTransform] = None,
                 num_samples: int = 1):
        super().__init__()
        data_dir = Path(data_dir)

        self.coils_path = data_dir / PROCESSED_ANTENNA_DIR_PATH / "antenna.h5"
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

        self.simulation_dir = data_dir / PROCESSED_SIMULATIONS_DIR_PATH
        self.simulation_list = self._get_simulations_list()

        ## TODO: check if transform valid:
        check_transforms(transforms)

        self.transforms = transforms

        if num_samples < 1:
            raise ValueError("The num_samples must be greater than 0")
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
        if not self.coils_path.exists():
            raise FileNotFoundError(f"File {self.coils_path} not found")
        with h5py.File(self.coils_path) as f:
            coils = f[ANTENNA_MASKS_OUT_KEY][:]
        return coils
    

    def _get_simulations_list(self) -> list:
        """
        This method searches for the list of `.h5` simulations files in the `simulations` directory.
        It also checks that the directory is not empty and throws an exception if it is so.

        Returns
        -------
        list
            List of simulation file paths 
        """
        simulations_list = natsorted(self.simulation_dir.glob("*.h5"))

        if len(simulations_list) == 0:
            raise FileNotFoundError(f"No simulations found in {self.simulation_dir}")
        
        return simulations_list

    
    @abstractmethod
    def _load_simulation(self, simulation_path: Union[Path, str]) -> DataItem:
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
    
    def _read_input(self, simulation_path: Union[Path, str]) -> npt.NDArray[np.float32]:
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
    
    def _get_dtype(self, simulation_path: Union[Path, str]) -> str:
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
    
    def _get_truncation_coefficients(self, simulation_path: Union[Path, str]) -> npt.NDArray:
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
