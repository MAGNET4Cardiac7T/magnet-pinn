"""
NAME
    grid.py
DESCRIPTION
    This module consists of the iterator of the voxelized electromagnetic simulation data, so it is in the 3d grid format.
"""
from typing import Union
from pathlib import Path

import numpy as np

from .dataitem import DataItem
from ._base import MagnetBaseIterator


class MagnetGridIterator(MagnetBaseIterator):
    """
    Iterator for loading the electromagnetic simulation data.
    """
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
            phase=np.zeros(self.num_coils),
            mask=np.ones(self.num_coils),
            coils=self.coils,
            dtype=self._get_dtype(simulation_path),
            truncation_coefficients=self._get_truncation_coefficients(simulation_path)
        )
