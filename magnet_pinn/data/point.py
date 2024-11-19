"""
NAME
    dataset.py
DESCRIPTION
    This module contains classes for loading the magnetostatic simulation data.
"""
import os
import h5py
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple, Union


import glob
import numpy as np
import pandas as pd
import numpy.typing as npt
from einops import reduce, pack, einsum, repeat, rearrange

import random
import torch

from magnet_pinn.preprocessing.preprocessing import (
    VOXEL_SIZE_OUT_KEY,
    ANTENNA_MASKS_OUT_KEY,
    MIN_EXTENT_OUT_KEY,
    MAX_EXTENT_OUT_KEY,
    FEATURES_OUT_KEY,
    E_FIELD_OUT_KEY,
    H_FIELD_OUT_KEY,
    COORDINATES_OUT_KEY,
    SUBJECT_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
    PROCESSED_ANTENNA_DIR_PATH,
    TRUNCATION_COEFFICIENTS_OUT_KEY,
    DTYPE_OUT_KEY
)


@dataclass
class DataItem:
    input: npt.NDArray[np.float32]
    subject: npt.NDArray[np.bool_]
    simulation: str
    field: Optional[npt.NDArray[np.float32]] = None
    positions: Optional[npt.NDArray[np.float32]] = None
    phase: Optional[npt.NDArray[np.float32]] = None
    mask: Optional[npt.NDArray[np.bool_]] = None
    coils: Optional[npt.NDArray[np.bool_]] = None
    dtype: Optional[str] = None,
    truncation_coefficients: Optional[npt.NDArray] = None


class MagnetPointIterator(torch.utils.data.IterableDataset):
    """
    Iterator for loading the electromagnetic simulation data for the point cloud models.
    """
    def __init__(self, 
                 data_dir: str,
                 point_samples_per_simulation: Union[int, float] = 0.1,
                 phase_samples_per_simulation: int = 10):
        super().__init__()
        self.simulation_dir = os.path.join(data_dir, PROCESSED_SIMULATIONS_DIR_PATH)
        self.coils_path = os.path.join(data_dir, PROCESSED_ANTENNA_DIR_PATH, "antenna.h5")
        self.simulation_list = glob.glob(os.path.join(self.simulation_dir, "*.h5"))
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

        self.num_points = self.coils.shape[0]
        if isinstance(point_samples_per_simulation, float):
            self.point_samples_per_simulation = int(self.num_points * point_samples_per_simulation)
        else:
            self.point_samples_per_simulation = point_samples_per_simulation
        self.phase_samples_per_simulation = phase_samples_per_simulation

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
            positions = f[COORDINATES_OUT_KEY][:]

            return DataItem(
                input=input_features,
                subject=np.max(subject, axis=-1),
                simulation=self._get_simulation_name(simulation_path),
                field=field,
                positions=positions,
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
            for i in range(self.phase_samples_per_simulation):
                augmented_simulation = self._augment_simulation(loaded_simulation, index=i)
                yield augmented_simulation.__dict__
    
    def __len__(self):
        return len(self.simulation_list)*self.phase_samples_per_simulation
    
    def _augment_simulation(self, simulation: DataItem, index: int = None) -> DataItem:
        """
        Method for augmenting the simulation data.
        Parameters
        ----------
        simulation : DataItem
            DataItem object with the simulation data
        
        Returns
        -------
        DataItem
            augmented DataItem object
        """
        phase, mask = self._sample_phase_and_mask(dtype=simulation.dtype, phase_index=index)
        point_indices = self._sample_point_indices()
        field_shifted = self._phase_shift_field(simulation.field, phase, mask, point_indices)
        coils_shifted = self._phase_shift_coils(phase, mask, point_indices)
        
        return DataItem(
            input=simulation.input[point_indices],
            subject=simulation.subject[point_indices],
            simulation=simulation.simulation,
            field=field_shifted,
            positions=simulation.positions[point_indices],
            phase=phase,
            mask=mask,
            coils=coils_shifted,
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients
        )
    
    def _sample_point_indices(self) -> npt.NDArray[np.int64]:
        return np.random.choice(self.num_points, self.point_samples_per_simulation, replace=False)
    
    def _sample_phase_and_mask(self, 
                               phase_index: int = None,
                               dtype: str = None
                               ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Method for sampling the phase and mask for the simulation.
        Parameters
        ----------
        phase_index : int
            Index of the phase sample
        
        Returns
        -------
        npt.NDArray[np.float32]:
            phase coefficients
        npt.NDArray[np.bool_]:
            mask for the phase coefficients
        """
        phase = np.random.uniform(0, 2*np.pi, self.num_coils)
        mask = np.random.choice([0, 1], self.num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], self.num_coils, replace=True)

        return phase.astype(dtype), mask.astype(np.bool_)    
    
    def _phase_shift_field(self, 
                           fields: npt.NDArray[np.float32], 
                           phase: npt.NDArray[np.float32], 
                           mask: npt.NDArray[np.float32], 
                           point_indices: npt.NDArray[np.int64]
                           ) -> npt.NDArray[np.float32]:
        fields = np.take(fields, point_indices, axis=2)
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((re_phase, im_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = repeat(coeffs, 'reimout reim coils -> he reimout reim coils', he=2)
        field_shift = einsum(fields, coeffs, 'he reim points fieldxyz coils, he reimout reim coils -> he reimout fieldxyz points')
        field_shift = rearrange(field_shift, 'he reimout fieldxyz points -> points fieldxyz reimout he')
        return field_shift

    def _phase_shift_coils(self,
                           phase: npt.NDArray[np.float32],
                           mask: npt.NDArray[np.bool_],
                           point_indices: npt.NDArray[np.int64]
                           ) -> npt.NDArray[np.float32]:
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs = np.stack((re_phase, im_phase), axis=0)
        coils_shift = einsum(self.coils[point_indices], coeffs, 'points coils, reim coils -> points reim')
        return coils_shift
