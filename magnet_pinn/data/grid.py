"""
NAME
    dataset.py
DESCRIPTION
    This module contains classes for loading the magnetostatic simulation data.
"""
from typing import Union
from pathlib import Path

import numpy as np

from .dataitem import DataItem
from ._base import MagnetBaseIterator


"""
Iterator for loading the magnetostatic simulation data.
"""
class MagnetGridIterator(MagnetBaseIterator):
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
            phase=np.zeros(self.num_coils),
            mask=np.ones(self.num_coils),
            coils=self.coils,
            dtype=self._get_dtype(simulation_path),
            truncation_coefficients=self._get_truncation_coefficients(simulation_path)
        )
