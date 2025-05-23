from abc import ABC

import trimesh
import numpy as np
from trimesh import Trimesh

from magnet_pinn.generator.meshes import Structure3D, Blob, Tube


class Serializer(ABC):
    def serialize(self, structure: Structure3D):
        raise NotImplementedError("Subclasses must implement `serialize` method")
    

class MeshSerializer(Serializer):
    def serialize(self, structure: Structure3D, subdivisions: int = 5) -> Trimesh:
        if isinstance(structure, Blob):
            return self._serialize_blob(structure, subdivisions)
        elif isinstance(structure, Tube):
            return self._serialize_tube(structure, subdivisions)
        else:
            raise ValueError("Unsupported structure type")
        

    def _serialize_blob(self, blob: Blob, subdivisions: int = 5) -> Trimesh:
        unit_sphere = trimesh.primitives.Sphere(1, subdivisions=subdivisions)
        offsets = blob.calculate_offsets(unit_sphere.vertices)
        vertices = (1 + offsets) * unit_sphere.vertices
        mesh = trimesh.Trimesh(vertices=vertices, faces=unit_sphere.faces)
        mesh.apply_translation(blob.position)
        mesh.apply_scale(blob.radius)
        return mesh
    
    def _serialize_tube(self, tube: Tube, subdivisions: int = 5) -> Trimesh:
        transform = (
            trimesh.transformations.translation_matrix(tube.position)
            @ trimesh.geometry.align_vectors([0, 0, 1], tube.direction)
        )
        return trimesh.creation.cylinder(
            radius=tube.radius,
            height=tube.height * tube.radius,
            sections=subdivisions ** 2,
            transform=transform
        )
