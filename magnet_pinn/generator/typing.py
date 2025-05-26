from typing import List
from dataclasses import dataclass

from trimesh import Trimesh


@dataclass
class PhantomItem:
    parent: Trimesh
    children: List[Trimesh]
    tubes: List[Trimesh]
