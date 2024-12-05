"""
NAME
    dataset.py
DESCRIPTION
    This module contains classes for loading the magnetostatic simulation data.
"""
from typing import Union
from pathlib import Path

import h5py
import numpy as np

from .dataitem import DataItem
from ._base import MagnetBaseIterator


from magnet_pinn.preprocessing.preprocessing import COORDINATES_OUT_KEY


class MagnetPointIterator(MagnetBaseIterator):

    def _load_simulation(self, simulation_path: Union[Path, str]) -> DataItem:
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
