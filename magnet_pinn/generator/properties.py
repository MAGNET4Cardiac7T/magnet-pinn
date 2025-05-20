"""
NAME
    properties.py
DESCRIPTION
    This module contains functions to generate physical properties for a mesh
"""
from typing import List

import numpy as np


class PropertySampler:
    def __init__(self, properties_cfg):
        self.properties_cfg = properties_cfg

    def sample(self, properties_list: List = None):
        if properties_list is None:
            properties_list = list(self.properties_cfg.keys())
        return {key: np.random.uniform(dim["min"], dim["max"]) for key, dim in self.properties_cfg.items() if key in properties_list}
