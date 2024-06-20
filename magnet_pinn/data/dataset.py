import os
import h5py

from typing import Any
import glob
import numpy as np
import pandas as pd


class MagnetBaseIterator:
    def __init__(self, 
                 data_dir: str) -> None:
        super().__init__()
        self.simulation_dir = os.path.join(data_dir, "simulations")
        self.coils_path = os.path.join(data_dir, "antenna", "antenna.h5")
        self.simulation_list = glob.glob(os.path.join(self.simulation_dir, "*.h5"))
        self.simulation_names = [os.path.basename(f)[:-3] for f in self.simulation_list]
        self.coils = self._read_coils()
        self.num_coils = self.coils.shape[-1]

    def _read_coils(self):
        with h5py.File(self.coils_path) as f:
            coils = f['masks'][:]
        return coils
    
    def __len__(self):
        return len(self.simulation_list)
    
    def _load_simulation(self, index: int):
        with h5py.File(self.simulation_list[index]) as f:
            item = {
                'input': f['input'][:],
                'subject': f['subject'][:],
                'efield': f['efield'][:],
                'hfield': f['hfield'][:],
                'simulation': self.simulation_names[index],
            }
        return item
    
    def __getitem__(self, index: int) -> Any:
        return self._load_simulation(index)
    
class PhaseAugmentedMagnetIterator(MagnetBaseIterator):
    def __init__(self, 
                 data_dir: str,
                 phase_samples_per_simulation: int = 100) -> None:
        super().__init__(data_dir)
        self.phase_samples_per_simulation = phase_samples_per_simulation

    def _sample_phase_and_mask(self, phase_index: int = None):
        phase = np.random.uniform(0, 2*np.pi, self.num_coils)
        mask = np.random.choice([0, 1], self.num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], self.num_coils, replace=True)

        return phase, mask    
    
    def _phase_shift_field(self, field, phase_coefficients):
        field = np.dot(field, phase_coefficients)
        field = np.concatenate([field.real, field.imag], axis=0)
        return field
    
    def __len__(self):
        return len(self.simulation_list)*self.phase_samples_per_simulation
    
    def __getitem__(self, index) -> Any:
        file_index = index // self.phase_samples_per_simulation
        phase_index = index % self.phase_samples_per_simulation

        item = self._load_simulation(file_index)
            
        phase, mask = self._sample_phase_and_mask(phase_index)

        item['coil_coefficients'] = np.exp(phase*1j)*mask
        item['coil_phase'] = phase
        item['coil_mask'] = mask

        item['coils_complex'] = np.dot(self.coils, item['coil_coefficients'])
        item['coils_real'] = np.stack([item['coils_complex'].real, item['coils_complex'].imag])

        # simulation phase shifter
        item['efield'] = self._phase_shift_field(item['efield'], item['coil_coefficients'])
        item['hfield'] = self._phase_shift_field(item['hfield'], item['coil_coefficients'])

        item = self.change_dtype(item, np.float32)

        return item
    
    def change_dtype(self, item, dtype):
        for key in item.keys():
            if isinstance(item[key], np.ndarray):
                if item[key].dtype == np.float64:
                    item[key] = item[key].astype(dtype)
        return item

class CoilEnumerationMagnetIterator(PhaseAugmentedMagnetIterator):
    def __init__(self, 
                 data_dir: str) -> None:
        super().__init__(data_dir)
        self.phase_samples_per_simulation = self.num_coils

    def _sample_phase_and_mask(self, phase_index: int = None):
        phase = np.zeros(self.num_coils)
        mask = np.zeros(self.num_coils)
        mask[phase_index] = 1
        
        phase = phase.astype(np.float32)
        mask = mask.astype(np.float32)
        return phase, mask
