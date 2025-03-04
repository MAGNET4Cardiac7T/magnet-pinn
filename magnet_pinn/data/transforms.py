"""
NAME
    transforms.py
DESCRIPTION
    This module contains classes for augmenting the simulation data.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Literal
from collections.abc import Iterable
from copy import copy

import numpy.typing as npt
import numpy as np
import einops

from .dataitem import DataItem


def check_transforms(transforms):
    if not isinstance(transforms, BaseTransform):
        raise ValueError("Transforms should be an instance of BaseTransform")
    elif isinstance(transforms, DefaultTransform):
        pass
    elif isinstance(transforms, Compose):
        if not sum(isinstance(t, PhaseShift) for t in transforms.transforms) == 1:
            raise ValueError("Exactly one of the composed transforms should be a PhaseShift transform")
    elif isinstance(transforms, PhaseShift):
        pass
    else:
        raise ValueError("Transforms not valid. Probably missing a phase shift transform")


class BaseTransform(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, simulation: DataItem):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.kwargs)
    
    def _check_data(self, simulation: DataItem):
        if not isinstance(simulation, DataItem):
            raise ValueError(f"Simulation should be an instance of DataItem, got {type(simulation)}")


class Compose(BaseTransform):
    """
    Compose function for combining multiple augmentations.

    Parameters
    ----------
    augmentations : List[BaseTransform]
        List of augmentations to be applied to the simulation data
    """
    def __init__(self, transforms: List[BaseTransform]):

        if not isinstance(transforms, Iterable):
            raise ValueError("Augmentations should be an iterable")
        elif len(list(transforms)) == 0:
            raise ValueError("No augmentations were given")
        else:
            for i in transforms:
                if i is None:
                    raise ValueError("Augmentation can not be None")
                elif not isinstance(i, BaseTransform):
                    raise ValueError(f"Augmentation should be an instance of BaseTransform, got {type(i)}")
        
        self.transforms = transforms

    def __call__(self, simulation: DataItem):
        """
        The method iterates over transformations and apply them to the simulation data.

        Parameters
        ----------
        simulation : DataItem
            simulation data

        Returns
        -------
        DataItem
            augmented simulation data
        """
        self._check_data(simulation)
        result = copy(simulation)
        for aug in self.transforms:
            result = aug(result)
        return result

    def __repr__(self):
        return self.__class__.__name__ + str(self.transforms)


class DefaultTransform(BaseTransform):
    """
    A default transform for the simulation data.
    It supposed to be used if the PhaseShift transform is not used.
    It changes field, coils, fully set phase to 0 and mask to 1.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, simulation: DataItem):
        """
        Sums the fields by coils, does same with coils data and define it as real part of the coils.
        Imaginary considered to be 0. Phase is set to 0 and mask to 1, both have size of the number of coils.

        Parameters
        ----------
        simulation : DataItem
            simulation data

        Returns
        -------
        DataItem
            augmented simulation data
        """
        self._check_data(simulation)

        num_coils = simulation.coils.shape[-1]
        field = np.sum(simulation.field, axis=-1)

        phase = np.zeros(num_coils, dtype=simulation.dtype)
        mask= np.ones(num_coils, dtype=np.bool_)

        coils_re = np.sum(simulation.coils, axis=-1)
        coils_im = np.zeros_like(coils_re)
        coils = np.stack((coils_re, coils_im), axis=0)

        return DataItem(
            input=simulation.input,
            subject=simulation.subject,
            simulation=simulation.simulation,
            field=field,
            phase=phase,
            mask=mask,
            coils=coils,
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients,
            positions=simulation.positions
        )
    
    def check_if_valid(self):
        return True


class Crop(BaseTransform):
    """
    Class for cropping the simulation data.

    Parameters
    ----------
    crop_size : Tuple[int, int, int]
        Size of the resulting data

    crop_position : Literal['random', 'center']
        Position of the crop
    """

    def __init__(self, 
                 crop_size: Tuple[int, int, int],
                 crop_position: Literal['random', 'center'] = 'center'):
        super().__init__()

        if not isinstance(crop_size, Iterable):
            raise ValueError("Crop size should be a tuple")
        elif len(crop_size) != 3:
            raise ValueError("Crop size should have 3 dimensional")
        elif not all((isinstance(i, int) for i in crop_size)):
            raise ValueError("Crop size should contain only integers")
        elif not (np.array(crop_size) > 0).all():
            raise ValueError("Crop size should be larger than 0")
        
        if crop_position not in ['random', 'center']:
            raise ValueError("Crop position should be either 'random' or 'center'")
        

        self.crop_size = crop_size
        self.crop_position = crop_position

    def __call__(self, simulation: DataItem):
        """
        Crops data based on the given arguments. In the case of the `center` crop position make almost equal margins on both sides.
        In the case of the `random` crop position, the crop margin is randomly selected as well as crop start. 
        Parameters
        ----------
        data : DataItem
            DataItem object with the simulation data
        
        Returns
        -------
        DataItem
            augmented DataItem object
        """
        self._check_data(simulation)
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
        for i in range(3):
            if crop_size[i] > full_size[i]:
                raise ValueError(f"crop size {crop_size} is larger than full size {full_size}")
        if full_size == crop_size:
            return (0, 0, 0)    
        elif self.crop_position == 'center':
            crop_start = [(full_size[i] - crop_size[i]) // 2 for i in range(3)]
        elif self.crop_position == 'random':
            crop_start = [np.random.randint(0, full_size[i] - crop_size[i]) for i in range(3)]
        else:
            raise ValueError(f"Unknown crop position {self.crop_position}")
        return crop_start
    

class PhaseShift(BaseTransform):
    def __init__(self, 
                 num_coils: int,
                 sampling_method: Literal['uniform', 'binomial'] = 'uniform'):
        super().__init__()

        if sampling_method not in ['uniform', 'binomial']:
            raise ValueError(f"Unknown masks sampling method {sampling_method}")

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

        self._check_data(simulation)

        phase, mask = self._sample_phase_and_mask(dtype=simulation.dtype)
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
            truncation_coefficients=simulation.truncation_coefficients,
            positions=simulation.positions
        )
    
    def _sample_phase_and_mask(self,
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
        phase = self._sample_phase(dtype)
        if self.sampling_method == 'uniform':
            mask = self._sample_mask_uniform()
        elif self.sampling_method == 'binomial':
            mask = self._sample_mask_binomial()
        else:
            raise ValueError(f"Unknown sampling method {self.sampling_method}")

        return phase.astype(dtype), mask.astype(np.bool_)  
    
    def _sample_phase(self, dtype: str = None) -> npt.NDArray[np.float32]:
        return np.random.uniform(0, 2*np.pi, self.num_coils).astype(dtype)
    
    def _sample_mask_uniform(self) -> npt.NDArray[np.bool_]:
        num_coils_on = np.random.randint(1, self.num_coils)
        mask = np.zeros(self.num_coils, dtype=bool)
        coils_on_indices = np.random.choice(self.num_coils, num_coils_on, replace=False)
        mask[coils_on_indices] = True
        return mask
    
    def _sample_mask_binomial(self) -> npt.NDArray[np.bool_]:
        mask = np.random.choice([0, 1], self.num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], self.num_coils, replace=True)
        return mask
    
    def _phase_shift_field(self, 
                           fields: npt.NDArray[np.float32], 
                           phase: npt.NDArray[np.float32], 
                           mask: npt.NDArray[np.float32], 
                           ) -> npt.NDArray[np.float32]:
        """
        Method of creating shift of field values. It uses a split formula for complex numbers multiplications.
        The calculations are done under the complex numbers in a split way.
        The shift formula is considered as 
        ```
        field_complex * (e ^ (phase * 1j)) * mask = field_complex * (cos(phase) + sin(phase) * 1j) * mask = (field_re * cos(phase) - field_im * sin(phase)) * mask + mask * (field_re * sin(phase) + field_im * cos(phase)) * 1j
        ```
        These coefficielnts are calculated for each coil and then it is summed up by the coils axis.
        """
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(fields, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift


    def _phase_shift_coils(self,
                           coils: npt.NDArray[np.float32],
                           phase: npt.NDArray[np.float32],
                           mask: npt.NDArray[np.bool_]
                           ) -> npt.NDArray[np.float32]:
        """
        Method of creation of shift values for the coils. It uses a split formula for complex numbers multiplications.
        ```
        coils * (e ^ (phase * 1j)) * mask = coils * (cos(phase) + sin(phase) * 1j) * mask = (coils_re * cos(phase) - coils_im * sin(phase)) * mask + mask * (coils_re * sin(phase) + coils_im * cos(phase)) * 1j
        ```
        It is also done for each coil and then summed up by the coils axis.
        """
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs = np.stack((re_phase, im_phase), axis=0)
        coils_shift = einops.einsum(coils, coeffs, '... coils, reim coils -> reim ...')
        return coils_shift
    

class GridPhaseShift(PhaseShift):
    """
    Class is added for the reversed comparability, the PhaseShift itself works fine with grid simulations
    """
    pass


## TODO move the coil enumerator logic to sample_phase and sample_mask methods, without modifying the _sample_phase_and_mask method
class CoilEnumeratorPhaseShift(PhaseShift):
    def __init__(self, 
                 num_coils: int):
        super().__init__(num_coils=num_coils)
        self.coil_on_index = 0

    
    def _sample_phase_and_mask(self, 
                               dtype: str = None
                               ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Augment simulation data by activating only a single coil in clockwise direction and setting phase to zero.
        Requires that num_samples == num_coils.

        ----------
        num_coils : int
            Number of coils in the simulation data
        dtype : str
            Data type of the phase coefficients
        
        Returns
        -------
        npt.NDArray[np.float32]:
            phase coefficients
        npt.NDArray[np.bool_]:
            mask for the phase coefficients
        """

        phase = self._sample_phase_zero(dtype=dtype)
        mask = self._sample_mask_single()

        return phase.astype(dtype), mask.astype(np.bool_)  
    
    def _sample_phase_zero(self, dtype: str = None) -> npt.NDArray[np.float32]:
        return np.zeros(self.num_coils).astype(dtype)
    
    def _sample_mask_single(self) -> npt.NDArray[np.bool_]:
        mask = np.zeros(self.num_coils, dtype=bool)
        mask[self.coil_on_index] = True
        self.coil_on_index = (self.coil_on_index + 1) % self.num_coils
        return mask

class PointPhaseShift(PhaseShift):
    """
    Class is added for the reversed comparability, the PhaseShift itself works fine with pointcloud simulations
    TODO: standartize the order od axis `fieldxyz` and `...`(meant x/y/z in grid and positions in pointcloud) in early 
    stages of simulation preprocessing
    """
    def _phase_shift_field(self, 
                           fields: npt.NDArray[np.float32], 
                           phase: npt.NDArray[np.float32], 
                           mask: npt.NDArray[np.float32], 
                           ) -> npt.NDArray[np.float32]:
        re_phase = np.cos(phase) * mask
        im_phase = np.sin(phase) * mask
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(fields, coeffs, 'hf reim ... fieldxyz coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift
    

class PointFeatureRearrange(BaseTransform):
    """
    Class for augmenting the simulation data.
    """
    def __init__(self, 
                 num_coils: int):
        super().__init__()
        self.num_coils = num_coils

    def __call__(self, simulation: DataItem):
        """
        This method changed the axis of the field and coils data.
        Parameters
        ----------
        simulation : DataItem
            DataItem object with the simulation data
        
        Returns
        -------
        DataItem
            augmented DataItem object
        """
        self._check_data(simulation)
        rearranged_field = self._rearrange_field(simulation.field, self.num_coils)
        rearranged_coils = self._rearrange_coils(simulation.coils, self.num_coils)
        return DataItem(
            input=simulation.input,
            subject=simulation.subject,
            simulation=simulation.simulation,
            field=rearranged_field,
            phase=simulation.phase,
            mask=simulation.mask,
            coils=rearranged_coils,
            dtype=simulation.dtype,
            truncation_coefficients=simulation.truncation_coefficients,
            positions=simulation.positions
        )

    def _rearrange_field(self, 
                         field: npt.NDArray[np.float32], 
                         num_coils: int) -> npt.NDArray[np.float32]:
        """
        Changes the order of data axis for the field data. The input data would have axis of 
        (field_type, re/im parts, x/y/z axis choice, values based on the position) and the output data would have change an order into 
        (position values, axis x/y/z choice, re/im parts, field_type).
        """
        field = einops.rearrange(field, 'he reimout fieldxyz points -> points fieldxyz reimout he')
        return field

    def _rearrange_coils(self, 
                         coils: npt.NDArray[np.float32], 
                         num_coils: int) -> npt.NDArray[np.float32]:
        """
        Changes the order of data axis for the coils data. The input data has a shape of 
        (re/im parts coils values for point) and the output data is the other way around
        """
        coils = einops.rearrange(coils, 'reim points -> points reim')
        return coils

class PointSampling(BaseTransform):
    """
    Class for sampling the points from the simulation data.
    Parameters
    ----------
    points_sampled : Union[float, int]
        Number of points to be sampled. If float, it is considered as a fraction of the total number of points.
    """
    def __init__(self, points_sampled: Union[float, int]):
        super().__init__()

        if not isinstance(points_sampled, (float, int)):
            raise ValueError("Points sampled should be either float or int")
        elif points_sampled <= 0:
            raise ValueError("The `points_sampled` parameter should be larger than 0")
        
        self.points_sampled = points_sampled

    def __call__(self, simulation: DataItem):
        self._check_data(simulation)
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
        """
        Main method for sampling the points from the simulation data. The `total_num_points` can be a percentage from the total number of it 
        and exact number of points to sample.

        Parameters
        ----------
        total_num_points : int
            Total number of points in the simulation data

        Returns
        -------
        npt.NDArray[np.int64]
            Indices of the sampled points
        """
        if isinstance(self.points_sampled, float):
            if self.points_sampled > 1.0:
                raise ValueError("In the case of the ratio sampling the ration should be less than 1.0")
            num_points_sampled = int(self.points_sampled * total_num_points)
        else:
            if self.points_sampled > total_num_points:
                raise ValueError("The number of points to sample should be less than the total number of points")
            num_points_sampled = self.points_sampled
        return np.random.choice(total_num_points, num_points_sampled, replace=False)
