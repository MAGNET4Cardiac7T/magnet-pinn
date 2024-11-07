"""
NAME
    dataset.py
DESCRIPTION
    This module contains classes for loading the magnetostatic simulation data.
"""
import os
import sys
import h5py
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple

import glob
import numpy as np
import pandas as pd
import numpy.typing as npt
from einops import reduce, pack

import queue
import threading
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
    re_efield: Optional[npt.NDArray[np.float32]] = None
    im_efield: Optional[npt.NDArray[np.float32]] = None
    re_hfield: Optional[npt.NDArray[np.float32]] = None
    im_hfield: Optional[npt.NDArray[np.float32]] = None
    re_phase: Optional[npt.NDArray[np.float32]] = None
    im_phase: Optional[npt.NDArray[np.float32]] = None
    phase: Optional[npt.NDArray[np.float32]] = None
    mask: Optional[npt.NDArray[np.bool_]] = None
    re_coils: Optional[npt.NDArray[np.float32]] = None
    im_coils: Optional[npt.NDArray[np.float32]] = None
    general_mask: Optional[npt.NDArray[np.bool_]] = None
    dtype: Optional[str] = None,
    truncation_coefficients: Optional[npt.NDArray] = None


class MagnetGridIterator(torch.utils.data.IterableDataset):
    """
    Iterator for loading the magnetostatic simulation data.
    """
    def __init__(self, 
                 data_dir: str,
                 phase_samples_per_simulation: int = 10):
        super().__init__()
        self.simulation_dir = os.path.join(data_dir, PROCESSED_SIMULATIONS_DIR_PATH)
        self.coils_path = os.path.join(data_dir, PROCESSED_ANTENNA_DIR_PATH, "antenna.h5")
        self.simulation_list = glob.glob(os.path.join(self.simulation_dir, "*.h5"))
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

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
            re_efield, im_efield = self._read_field(f, E_FIELD_OUT_KEY)
            re_hfield, im_hfield = self._read_field(f, H_FIELD_OUT_KEY)
            input_features = self._read_input_features(f, FEATURES_OUT_KEY, TRUNCATION_COEFFICIENTS_OUT_KEY)
            subject = self._read_subject(f, SUBJECT_OUT_KEY)
            coils = self.coils[:]
            stacked_arr, _ = pack(
                [subject, coils],
                "x y z *"
            )
            stacked_arr = np.ascontiguousarray(stacked_arr)
            general_mask = np.ascontiguousarray(reduce(
                stacked_arr,
                "x y z c -> x y z",
                "max"
            )).astype(np.bool_)

            return DataItem(
                input=input_features,
                subject=np.max(subject, axis=-1),
                simulation=self._get_simulation_name(simulation_path),
                re_efield=re_efield,
                im_efield=im_efield,
                re_hfield=re_hfield,
                im_hfield=im_hfield,
                general_mask=general_mask,
                dtype=f.attrs[DTYPE_OUT_KEY],
                truncation_coefficients=f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY]
            )
        

    def _read_field(self, f: h5py.File, field_key: str) -> Dict:
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
        field_val = f[field_key][:] 
        if field_val.dtype.names is None:
            return field_val.real, field_val.imag
        
        return field_val["re"], field_val["im"]
    

    def _read_input_features(self, f: h5py.File, features_key: str, truncation_key: str) -> npt.NDArray[np.float32]:
        """
        Method for reading the input features from the h5 file.

        Parameters
        ----------
        f : h5py.File
            h5 file descriptor
        features_key : str
            key for the features
        truncation_key : str
            key for the truncation coefficients

        Returns
        -------
        npt.NDArray[np.float32]
            input features array
        """
        features = f[features_key][:]
        return features
    
    def _read_subject(self, f: h5py.File, subject_key: str) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int8]]:
        """
        Method for reading the subject from the h5 file.

        Parameters
        ----------
        f : h5py.File
            h5 file descriptor
        subject_key : str
            key for the subject

        Returns
        -------
        Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int8]]
            subject array as bool (one-hot) and subject integer array (as labels)
        """
        subject = f[subject_key][:]
        return subject
    
    def __iter__(self):
        random.shuffle(self.simulation_list)
        for simulation in self.simulation_list:
            for i in range(self.phase_samples_per_simulation):
                loaded_simulation = self._load_simulation(simulation)
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
        re_efield_shift, im_efield_shift = self._phase_shift_field(
            simulation.re_efield, simulation.im_efield, phase, mask
        )
        re_hfield_shift, im_hfield_shift = self._phase_shift_field(
            simulation.re_hfield, simulation.im_hfield, phase, mask
        )

        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask

        re_coils = self.coils * re_phase
        im_coils = self.coils * im_phase
        
        return DataItem(
            input=simulation.input,
            subject=simulation.subject,
            simulation=simulation.simulation,
            re_efield=re_efield_shift,
            im_efield=im_efield_shift,
            re_hfield=re_hfield_shift,
            im_hfield=im_hfield_shift,
            re_phase=re_phase,
            im_phase=im_phase,
            phase=phase,
            mask=mask,
            re_coils=re_coils,
            im_coils=im_coils,
            general_mask=simulation.general_mask,
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients
        )
    
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
                           re_field: npt.NDArray[np.float32], 
                           im_field: npt.NDArray[np.float32], 
                           re_phase: npt.NDArray[np.float32], 
                           im_phase: npt.NDArray[np.float32]
                           ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        re_field_shift = re_field * re_phase - im_field * im_phase
        im_field_shift = re_field * re_phase + im_field * im_phase
        
        return re_field_shift, im_field_shift