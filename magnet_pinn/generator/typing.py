"""
NAME
    typing.py

DESCRIPTION
    This module provides type definitions and data structures for phantom generation.
    Contains type-safe data classes and type aliases for representing phantoms at
    different stages of the generation pipeline, from abstract structures to
    concrete meshes with material properties.
"""
from dataclasses import dataclass
from typing import List, Union, Tuple

from trimesh import Trimesh

from .structures import Structure3D

Point3D = Tuple[float, float, float]
FaceIndices = Tuple[int, ...]
MeshGrid = List[List[Tuple[float, float, float]]]


@dataclass
class PropertyItem:
    """Material properties for a single phantom component."""
    conductivity: float
    permittivity: float
    density: float


@dataclass
class StructurePhantom:
    """Phantom representation using abstract 3D geometric structures."""
    parent: Structure3D
    children: List[Structure3D]
    tubes: List[Structure3D]


@dataclass
class MeshPhantom:
    """Phantom representation using concrete triangular mesh objects."""
    parent: Trimesh
    children: List[Trimesh]
    tubes: List[Trimesh]


@dataclass
class PropertyPhantom:
    """Phantom representation with material properties for each component."""
    parent: PropertyItem
    children: List[PropertyItem]
    tubes: List[PropertyItem]


PhantomItem = Union[StructurePhantom, MeshPhantom, PropertyPhantom]
