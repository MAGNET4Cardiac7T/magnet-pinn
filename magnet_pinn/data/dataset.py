import os
import h5py
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import glob
import numpy as np
import pandas as pd
import numpy.typing as npt


OLD_LOWER_BOUND = np.array(
    [-240, -150, -230]
)
OLD_UPPER_BOUND = np.array(
    [240, 150, 150]
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


class MagnetBaseIterator:

    crop_mask: Optional[Tuple] = None

    def __init__(self, 
                 data_dir: str,
                 crop_data: bool = False,
                 crop_left_shift: bool = True
                 ) -> None:
        super().__init__()
        self.simulation_dir = os.path.join(data_dir, "simulations")
        self.coils_path = os.path.join(data_dir, "antenna", "antenna.h5")
        self.simulation_list = glob.glob(os.path.join(self.simulation_dir, "*.h5"))
        self.simulation_names = [os.path.basename(f)[:-3] for f in self.simulation_list]
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

        self.crop_data = crop_data
        self.crop_left_shift = crop_left_shift
        self._set_crop_mask()

    def _read_coils(self):
        with h5py.File(self.coils_path) as f:
            coils = f['masks'][:]
        return coils
    
    def _set_crop_mask(self):
        with h5py.File(self.simulation_list[0]) as f:
            min_extent = f.attrs["min_extent"]
            max_extent = f.attrs["max_extent"]
            voxel_size = f.attrs["voxel_size"]
            shape = f["efield"][:].shape[1:-1]

        if not self.crop_data:
            self.crop_mask = (
                slice(0, shape[0]),
                slice(0, shape[1]),
                slice(0, shape[2])
            )
            return

        left_shift = (OLD_LOWER_BOUND - min_extent) % voxel_size
        right_shift = (max_extent - OLD_UPPER_BOUND) % voxel_size

        shifted_old_lower_bound = OLD_LOWER_BOUND - left_shift * self.crop_left_shift + left_shift * (1 - self.crop_left_shift)
        shifted_old_upper_bound = OLD_UPPER_BOUND - right_shift * self.crop_left_shift + right_shift * (1 - self.crop_left_shift)

        lower_bound = ((shifted_old_lower_bound - min_extent) // voxel_size).astype(int)
        upper_bound = shape - ((max_extent - shifted_old_upper_bound) // voxel_size).astype(int)

        self.crop_mask = (
            slice(lower_bound[0], upper_bound[0]),
            slice(lower_bound[1], upper_bound[1]),
            slice(lower_bound[2], upper_bound[2])
        )
    
    def __len__(self):
        return len(self.simulation_list)
    
    def _load_simulation(self, index: int) -> DataItem:
        with h5py.File(self.simulation_list[index]) as f:
            re_efield, im_efield = self._read_field(f, 'efield')
            re_hfield, im_hfield = self._read_field(f, 'hfield')

            return DataItem(
                input=f['input'][:, *self.crop_mask].astype(np.float32),
                subject=f['subject'][*self.crop_mask, :].astype(np.bool_),
                simulation=self.simulation_names[index],
                re_efield=re_efield,
                im_efield=im_efield,
                re_hfield=re_hfield,
                im_hfield=im_hfield
            )
    
    def _read_field(self, f: h5py.File, field_key: str) -> Dict:
        """
        A method for reading the field from the h5 file.
        Reads and splits the field into real and imaginary parts.

        Parameters
        ----------
        f : h5py.File
            h5 file descriptor
        field_key : str
            field database key

        Returns
        -------
        Dict
            A dictionary with `re_field_key` and `im_field_key` keys
            with real and imaginary parts of the field
        """
        field_val = f[field_key][:, *self.crop_mask, :] 
        if field_val.dtype.names is None:
            return field_val.real.astype(np.float32), field_val.imag.astype(np.float32)
        
        return field_val["re"].astype(np.float32), field_val["im"].astype(np.float32)
    
    def __getitem__(self, index: int) -> Any:
        return self._load_simulation(index)


class PhaseAugmentedMagnetIterator(MagnetBaseIterator):
    def __init__(self, 
                 data_dir: str,
                 phase_samples_per_simulation: int = 100, *args, **kwargs) -> None:
        super().__init__(data_dir, *args, **kwargs)
        self.phase_samples_per_simulation = phase_samples_per_simulation

    def _sample_phase_and_mask(self, phase_index: int = None):
        phase = np.random.uniform(0, 2*np.pi, self.num_coils)
        mask = np.random.choice([0, 1], self.num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], self.num_coils, replace=True)

        return phase.astype(np.float32), mask.astype(np.bool_)    
    
    def _phase_shift_field(self, 
                           re_field: npt.NDArray[np.float32], 
                           im_field: npt.NDArray[np.float32], 
                           re_phase: npt.NDArray[np.float32], 
                           im_phase: npt.NDArray[np.float32]
                           ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        re_field_shift = re_field * re_phase - im_field * im_phase
        im_field_shift = re_field * re_phase + im_field * im_phase
        
        return re_field_shift, im_field_shift
    
    def __len__(self):
        return len(self.simulation_list)*self.phase_samples_per_simulation
    
    def __getitem__(self, index) -> Any:
        file_index = index // self.phase_samples_per_simulation
        phase_index = index % self.phase_samples_per_simulation

        item = self._load_simulation(file_index)
            
        item.phase, item.mask = self._sample_phase_and_mask(phase_index)

        # e^(i * phase) * mask = cos(phase) * mask + i * sin(phase) * mask
        item.re_phase = np.cos(item.phase) * item.mask
        item.im_phase = np.sin(item.phase) * item.mask

        item.re_coils = self.coils * item.re_phase
        item.im_coils = self.coils * item.im_phase

        # split dot product of complex numbers array
        item.re_efield = item.re_efield * item.re_phase - item.im_efield * item.im_phase
        item.im_efield = item.re_efield * item.re_phase + item.im_efield * item.im_phase

        item.re_hfield = item.re_hfield * item.re_phase - item.im_hfield * item.im_phase
        item.im_hfield = item.re_hfield * item.re_phase + item.im_hfield * item.re_phase
        return item


class CoilEnumerationMagnetIterator(PhaseAugmentedMagnetIterator):
    def __init__(self, 
                 data_dir: str, *args, **kwargs) -> None:
        super().__init__(data_dir, *args, **kwargs)
        self.phase_samples_per_simulation = self.num_coils

    def _sample_phase_and_mask(self, phase_index: int = None):
        phase = np.zeros(self.num_coils)
        mask = np.zeros(self.num_coils)
        mask[phase_index] = 1
        
        phase = phase.astype(np.float32)
        mask = mask.astype(np.float32)
        return phase, mask
