from typing import List
from dataclasses import dataclass

from trimesh import Trimesh

from .structures import Structure3D


@dataclass
class PropertyItem:
    conductivity: float
    permittivity: float
    density: float


@dataclass
class PhantomItem:
    parent: Trimesh | Structure3D | PropertyItem
    children: List[Trimesh | Structure3D | PropertyItem]
    tubes: List[Trimesh | Structure3D | PropertyItem]
