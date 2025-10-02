"""
NAME
    _base.py
DESCRIPTION
    This module consists of the abstract base class for loading the electromagnetic simulation data.
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
from .transforms import BaseTransform, check_transforms

from magnet_pinn.preprocessing.preprocessing import (
    ANTENNA_MASKS_OUT_KEY,
    FEATURES_OUT_KEY,
    E_FIELD_OUT_KEY,
    H_FIELD_OUT_KEY,
    SUBJECT_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
    PROCESSED_ANTENNA_DIR_PATH,
    TRUNCATION_COEFFICIENTS_OUT_KEY,
    DTYPE_OUT_KEY,
    COORDINATES_OUT_KEY
)


class MagnetBaseIterator(torch.utils.data.IterableDataset, ABC):
    """
    Abstract base Iterator class for loading the electromagnetic simulation data.

    Parameters
    ----------
    data_dir : Union[str, Path]
        A data directory, which was created after the preprocessing step
    transforms : Optional[BaseTransform]
        Transformations to apply to the data during the data loading, can have a sequence of transformations,
        at least one of them should make a phase shift of the field
    num_samples : int
        Number of samples to generate from each simulation

    Attributes
    ----------
    coils_path : Union[str, Path]
        Path to the file with the coils masks
    simulation_dir : Union[str, Path]
        Path to the directory with the simulations
    transforms : Optional[BaseTransform]
        Transformations to apply to the data during the data loading, can have a sequence of transformations,
        at least one of them should make a phase shift of the field
    num_samples : int
        Number of samples to generate from each simulation
    coils: npt.NDArray[np.bool_]
        Coils masks array
    num_coils: int
        Number of coils
    simulation_list: List[Path]
        List of simulation `.h5` file paths
    """
    def __init__(self, 
                 data_dir: Union[str, Path],
                 transforms: Optional[BaseTransform] = None,
                 num_samples: int = 1):
        """
        Parameters
        ----------
        data_dir : Union[str, Path]
            A data directory, which was created after the preprocessing step
        transforms : Optional[BaseTransform]
            Transformations to apply to the data during the data loading, can have a sequence of transformations,
            at least one of them should make a phase shift of the field
        num_samples : int
            Number of samples to generate from each simulation
        """
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

    def _get_simulation_name(self, simulation_path: Union[str, Path]) -> str:
        """
        Method gets the simulation file name without an extension `.h5`.

        Parameters
        ----------
        simulation_path : Union[str, Path]
            Path to the simulation file

        Returns
        -------
        str
            File name without an extension
        """
        return os.path.basename(simulation_path)[:-3]

    def _read_coils(self) -> npt.NDArray[np.bool_]:
        """
        Method reads coils masks from the h5 file and returns it as a float array.

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
    
    def _load_simulation(self, simulation_path: Union[Path, str]) -> DataItem:
        """
        Main method to implement for the children of the `MagnetBaseIterator` class.
        It loads the data from the simulation file and return the `DataItem` object.
        
        Parameters
        ----------
        simulation_path : Union[Path, str]
            Path to the simulation file
        
        Returns
        -------
        DataItem
            DataItem object with the loaded data
        """
        return DataItem(
            input=self._read_input(simulation_path),
            subject=self._read_subject(simulation_path),
            simulation=self._get_simulation_name(simulation_path),
            field=self._read_fields(simulation_path),
            positions=self._read_positions(simulation_path),
            phase=np.zeros(self.num_coils),
            mask=np.ones(self.num_coils),
            coils=self.coils,
            dtype=self._get_dtype(simulation_path),
            truncation_coefficients=self._get_truncation_coefficients(simulation_path)
        )    

    def _read_fields(self, simulation_path: Union[str, Path]) -> npt.NDArray[np.float32]:
        """
        A method for reading the field from the h5 file. After the extraction 
        we join e- and h-fields in the first axis, and real and imaginary parts in the second axis.
        As a result we get the axis (e/h, re/im, ...). The next axis are dependent on the grid/pointscloud
        type of the data.

        Parameters
        ----------
        simulation_path : Union[str, Path]
            Path to the simulation

        Returns
        -------
        npy.NDArray[np.float32]
            Field array
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

        Parameters
        ----------
        simulation_path : Union[Path, str]
            Path to the simulation

        Returns
        -------
        npt.NDArray[np.float32]
            Input features array
        """
        with h5py.File(simulation_path) as f:
            features = f[FEATURES_OUT_KEY][:]
        return features
    
    def _read_subject(self, simulation_path: Union[str, Path]) -> npt.NDArray[np.bool_]:
        """
        Method reads the subject mask from the h5 file.

        Parameters
        ----------
        simulation_path : Union[str, Path]
            Path to the simulation

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

        Parameters
        ----------
        simulation_path : Union[Path, str]
            Path to the simulation

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

        Parameters
        ----------
        simulation_path : Union[Path, str]
            Path to the simulation

        Returns
        -------
        npt.NDArray
            Truncation coefficients
        """
        with h5py.File(simulation_path) as f:
            truncation_coefficients = f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY]
        return truncation_coefficients

    def _read_positions(self, simulation_path: str) -> np.ndarray:
        """
        Reads the positions of points from the h5 file. 
        Parameters
        ----------
        simulation_path : str
            Path to the simulation file

        Returns 
        -------
        np.ndarray
            Positions of points
        """

        with h5py.File(simulation_path, 'r') as f:
            positions = f[COORDINATES_OUT_KEY][:]
        return positions
    
    def __iter__(self):
        """
        The main method to iterate. It shuffles the simulation list and then for each simulation
        it loads the data, applies the transformations and yields the augmented data.
        """
        random.shuffle(self.simulation_list)
        for simulation in self.simulation_list:
            loaded_simulation = self._load_simulation(simulation)
            for i in range(self.num_samples):
                augmented_simulation = self.transforms(loaded_simulation)
                yield augmented_simulation.__dict__
    
    def __len__(self):
        return len(self.simulation_list)*self.num_samples
