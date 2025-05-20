from abc import ABC

import trimesh
import numpy as np
from trimesh import Trimesh

from magnet_pinn.generator.meshes import Structure3D, Blob, Tube


class Serializer(ABC):
    def serialize(self, structure: Structure3D):
        raise NotImplementedError("Subclasses must implement `serialize` method")
    

class TrimeshSerializer(Serializer):
    def serialize(self, structure: Structure3D) -> Trimesh:
        if isinstance(structure, Blob):
            return Trimesh(vertices=structure.vertices, faces=structure.faces)
        elif isinstance(structure, Tube):
            return trimesh.creation.cylinder(
                radius=structure.radius,
                segment=np.vstack([structure.start, structure.end]),
                sections=structure.subdivisions ** 2,
                transform=structure.transform
            )
        else:
            raise ValueError("Unsupported structure type")
