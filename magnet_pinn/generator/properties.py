"""
NAME
    properties.py

DESCRIPTION
    This module provides physical property sampling for mesh phantoms.
    Contains utilities for generating realistic material properties (conductivity, 
    permittivity, density) that can be assigned to mesh components in MRI simulations.
"""
from typing import List, Union

import numpy as np

from .typing import PropertyItem, PropertyPhantom, MeshPhantom, StructurePhantom


class PropertySampler:
    """
    Sampler for generating physical properties of phantom components.
    
    Randomly samples material properties from configured distributions for each
    component of a phantom structure, enabling realistic material property
    assignment for electromagnetic simulations.
    """
    def __init__(self, properties_cfg):
        """
        Initialize the property sampler with configuration parameters.

        Parameters
        ----------
        properties_cfg : dict
            Configuration dictionary specifying property ranges.
            Each key should be a property name (e.g., 'conductivity') with 
            a value dict containing 'min' and 'max' keys for range bounds.
        """
        self.properties_cfg = properties_cfg

    def sample_like(self, item: Union[StructurePhantom, MeshPhantom], properties_list: List = None) -> PropertyPhantom:
        """
        Sample material properties for all components of a phantom structure.

        Parameters
        ----------
        item : Union[StructurePhantom, MeshPhantom]
            The phantom structure to sample properties for. Must have parent,
            children, and tubes attributes.
        properties_list : List, optional
            List of property names to sample. If None, samples all configured properties.

        Returns
        -------
        PropertyPhantom
            A phantom with sampled material properties for all components.
        """
        return PropertyPhantom(
            parent=self._sample(properties_list),
            children=[self._sample(properties_list) for _ in item.children],
            tubes=[self._sample(properties_list) for _ in item.tubes]
        )

    def _sample(self, properties_list: List = None):
        """
        Sample a single set of material properties.

        Parameters
        ----------
        properties_list : List, optional
            List of property names to sample. If None, samples all configured properties.

        Returns
        -------
        PropertyItem
            A single property item with randomly sampled values.
        """
        if properties_list is None:
            properties_list = list(self.properties_cfg.keys())
        return PropertyItem(**{
            key: np.random.uniform(dim["min"], dim["max"]) 
            for key, dim in self.properties_cfg.items() 
            if key in properties_list
        })
