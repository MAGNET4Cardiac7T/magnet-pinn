"""
NAME
    properties.py
DESCRIPTION
    This module contains functions to generate physical properties for a mesh
"""
from typing import List

import numpy as np

from .typing import PropertyItem, PhantomItem


class PropertySampler:
    def __init__(self, properties_cfg):
        self.properties_cfg = properties_cfg

    def sample_like(self, item: PhantomItem, properties_list: List = None) -> PropertyItem:
        return PhantomItem(
            parent=self._sample(properties_list),
            children=[self._sample(properties_list) for _ in item.children],
            tubes=[self._sample(properties_list) for _ in item.tubes]
        )

    def _sample(self, properties_list: List = None):
        if properties_list is None:
            properties_list = list(self.properties_cfg.keys())
        return PropertyItem(**{
            key: np.random.uniform(dim["min"], dim["max"]) 
            for key, dim in self.properties_cfg.items() 
            if key in properties_list
        })
