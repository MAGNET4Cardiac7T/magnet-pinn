"""
    A module for preprocessing data for the magnet_pinn package.
"""

from .preprocessing import GridPreprocessing, PointPreprocessing
from .voxelizing_mesh import MeshVoxelizer
from .simulation import Simulation
from .reading_properties import PropertyReader
from .reading_field import FieldReaderFactory, GridReader, PointReader

__all__ = [
    'Preprocessing', 
    'GridPreprocessing',
    'PointPreprocessing',
    'Simulation',
    'MeshVoxelizer',
    'PropertyReader',
    'FieldReaderFactory',
    'FieldReader',
    'GridReader',
    'PointReader',
]