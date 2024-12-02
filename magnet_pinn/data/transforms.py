from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Literal
import numpy.typing as npt
import numpy as np
import einops

from .dataitem import DataItem



class BaseTransform(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, simulation: DataItem):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.kwargs)
    
class Compose(BaseTransform):
    def __init__(self, augmentations: List[BaseTransform]):
        self.augmentations = augmentations

    def __call__(self, simulation: DataItem):
        for aug in self.augmentations:
            simulation = aug(simulation)
        return simulation

    def __repr__(self):
        return self.__class__.__name__ + str(self.augmentations)
    


class Crop(BaseTransform):
    def __init__(self, crop_size: Tuple[int, int, int]):
        super().__init__()
        self.crop_size = crop_size

    def __call__(self, simulation: DataItem):
        """
        Method for augmenting the simulation data.
        Parameters
        ----------
        data : DataItem
            DataItem object with the simulation data
        
        Returns
        -------
        DataItem
            augmented DataItem object
        """
        crop_size = self.crop_size
        full_size = simulation.input.shape[1:]
        crop_start = self._sample_crop_start(full_size, crop_size)
        crop_mask = tuple(slice(crop_start[i], crop_start[i] + crop_size[i]) for i in range(3))

        return DataItem(
            input=self._crop_array(simulation.input, crop_mask, 1),
            subject=self._crop_array(simulation.subject, crop_mask, 0),
            simulation=simulation.simulation,
            field=self._crop_array(simulation.field, crop_mask, 3),
            phase=simulation.phase,
            mask=simulation.mask,
            coils=self._crop_array(simulation.coils, crop_mask, 0),
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients,
        )
    
    def _crop_array(self, 
                    array: npt.NDArray[np.float32],
                    crop_mask: Tuple[slice, slice, slice], 
                    starting_axis: int) -> npt.NDArray[np.float32]:
        crop_mask = (slice(None), )*starting_axis + crop_mask
        return array[*crop_mask]

    
    def _sample_crop_start(self, full_size: Tuple[int, int, int], crop_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        crop_start = [np.random.randint(0, full_size[i] - crop_size[i]) for i in range(3)]
        return crop_start

    

class PhaseShift(BaseTransform):
    def __init__(self, 
                 num_coils: int,
                 sampling_method: Literal['uniform', 'binomial'] = 'uniform'):
        super().__init__()
        self.num_coils = num_coils
        self.sampling_method = sampling_method

    def __call__(self, simulation: DataItem):
        """
        Method for augmenting the simulation data.
        Parameters
        ----------
        data : DataItem
            DataItem object with the simulation data
        
        Returns
        -------
        DataItem
            augmented DataItem object
        """

        phase, mask = self._sample_phase_and_mask(dtype=simulation.dtype, num_coils=self.num_coils)
        field_shifted = self._phase_shift_field(simulation.field, phase, mask)
        coils_shifted = self._phase_shift_coils(simulation.coils, phase, mask)
        
        return DataItem(
            input=simulation.input,
            subject=simulation.subject,
            simulation=simulation.simulation,
            field=field_shifted,
            phase=phase,
            mask=mask,
            coils=coils_shifted,
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients
        )
    
    def _sample_phase_and_mask(self, 
                               num_coils: int,
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
        phase = self._sample_phase(num_coils, dtype)
        if self.sampling_method == 'uniform':
            mask = self._sample_mask_uniform(num_coils)
        elif self.sampling_method == 'binomial':
            mask = self._sample_mask_binomial(num_coils)
        else:
            raise ValueError(f"Unknown sampling method {self.sampling_method}")

        return phase.astype(dtype), mask.astype(np.bool_)  
    
    def _sample_phase(self, num_coils: int, dtype: str = None) -> npt.NDArray[np.float32]:
        return np.random.uniform(0, 2*np.pi, num_coils).astype(dtype)
    
    def _sample_mask_uniform(self, num_coils: int) -> npt.NDArray[np.bool_]:
        num_coils_on = np.random.randint(1, num_coils)
        mask = np.zeros(num_coils, dtype=bool)
        coils_on_indices = np.random.choice(num_coils, num_coils_on, replace=False)
        mask[coils_on_indices] = True
        return mask
    
    def _sample_mask_binomial(self, num_coils: int) -> npt.NDArray[np.bool_]:
        mask = np.random.choice([0, 1], num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], num_coils, replace=True)
        return mask
    
    @abstractmethod
    def _phase_shift_field(self, 
                           fields: npt.NDArray[np.float32], 
                           phase: npt.NDArray[np.float32], 
                           mask: npt.NDArray[np.float32], 
                           ) -> npt.NDArray[np.float32]:
        raise NotImplementedError
    
    @abstractmethod
    def _phase_shift_coils(self,
                           coils: npt.NDArray[np.float32],
                           phase: npt.NDArray[np.float32],
                           mask: npt.NDArray[np.bool_]
                           ) -> npt.NDArray[np.float32]:
        raise NotImplementedError


class GridPhaseShift(PhaseShift):
    def __init__(self, num_coils: int):
        super().__init__(num_coils=num_coils)

    def _phase_shift_field(self, 
                           fields: npt.NDArray[np.float32], 
                           phase: npt.NDArray[np.float32], 
                           mask: npt.NDArray[np.float32], 
                           ) -> npt.NDArray[np.float32]:
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((re_phase, im_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(fields, coeffs, 'hf reim fieldxyz x y z coils, hf reimout reim coils -> hf reimout fieldxyz x y z')
        return field_shift


    def _phase_shift_coils(self,
                           coils: npt.NDArray[np.float32],
                           phase: npt.NDArray[np.float32],
                           mask: npt.NDArray[np.bool_]
                           ) -> npt.NDArray[np.float32]:
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs = np.stack((re_phase, im_phase), axis=0)
        coils_shift = einops.einsum(coils, coeffs, 'x y z coils, reim coils -> reim x y z')
        return coils_shift
    

class PointPhaseShift(PhaseShift):
    def __init__(self, num_coils: int):
        super().__init__(num_coils=num_coils)

    def _phase_shift_field(self, 
                           fields: npt.NDArray[np.float32], 
                           phase: npt.NDArray[np.float32], 
                           mask: npt.NDArray[np.float32]
                           ) -> npt.NDArray[np.float32]:
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((re_phase, im_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> he reimout reim coils', he=2)
        field_shift = einops.einsum(fields, coeffs, 'he reim fieldxyz points coils, he reimout reim coils -> he reimout fieldxyz points')
        field_shift = einops.rearrange(field_shift, 'he reimout fieldxyz points -> points fieldxyz reimout he')
        return field_shift

    def _phase_shift_coils(self,
                           coils: npt.NDArray[np.float32],
                           phase: npt.NDArray[np.float32],
                           mask: npt.NDArray[np.bool_]
                           ) -> npt.NDArray[np.float32]:
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs = np.stack((re_phase, im_phase), axis=0)
        coils_shift = einops.einsum(coils, coeffs, 'points coils, reim coils -> points reim')
        return coils_shift
    

class PointSampling(BaseTransform):
    def __init__(self, points_sampled: Union[float, int]):
        super().__init__()
        self.points_sampled = points_sampled

    def __call__(self, simulation: DataItem):
        total_num_points = simulation.positions.shape[0]
        point_indices = self._sample_point_indices(total_num_points=total_num_points)
        return DataItem(
            input=simulation.input[point_indices],
            subject=simulation.subject[point_indices],
            simulation=simulation.simulation,
            field=simulation.field[:, :, point_indices],
            phase=simulation.phase,
            mask=simulation.mask,
            coils=simulation.coils[point_indices],
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients,
            positions=simulation.positions[point_indices]
        )

    def _sample_point_indices(self, total_num_points: int) -> npt.NDArray[np.int64]:
        if isinstance(self.points_sampled, float):
            num_points_sampled = int(self.points_sampled * total_num_points)
        else:
            num_points_sampled = self.points_sampled
        return np.random.choice(total_num_points, num_points_sampled, replace=False)