"""
NAME
    serializers.py

DESCRIPTION
    This module provides serialization of geometric structures to mesh representations.
    Contains serializers for converting abstract 3D structures (blobs, tubes) into
    concrete triangular mesh objects suitable for visualization and numerical simulation.
"""
from abc import ABC

import trimesh
import numpy as np
from trimesh import Trimesh

from .structures import Structure3D, Blob, Tube, CustomMeshStructure


class Serializer(ABC):
    """
    Abstract base class for structure-to-mesh serialization.
    
    Defines the interface for converting abstract geometric structures
    into concrete mesh representations for computational processing.
    """
    def serialize(self, structure: Structure3D):
        """
        Serialize a 3D structure to mesh representation.

        Parameters
        ----------
        structure : Structure3D
            The 3D structure to convert to mesh format.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement `serialize` method")
    

class MeshSerializer(Serializer):
    """
    Serializer for converting 3D structures to triangular mesh representations.
    
    Transforms geometric structures (blobs with Perlin noise deformation, tubes)
    into high-quality triangular meshes with configurable subdivision levels
    for optimal simulation accuracy and performance.
    """
    def serialize(self, structure: Structure3D, subdivisions: int = 5) -> Trimesh:
        """
        Convert a 3D structure to a triangular mesh.

        Parameters
        ----------
        structure : Structure3D
            The structure to serialize (Blob or Tube).
        subdivisions : int, optional
            Number of subdivisions for mesh quality. Default is 5.
            Higher values create smoother meshes with more vertices.

        Returns
        -------
        Trimesh
            Triangular mesh representation of the input structure.

        Raises
        ------
        ValueError
            If the structure type is not supported.
        """
        if isinstance(structure, Blob):
            return self._serialize_blob(structure, subdivisions)
        elif isinstance(structure, Tube):
            return self._serialize_tube(structure, subdivisions)
        elif isinstance(structure, CustomMeshStructure):
            return structure.mesh.copy()
        else:
            raise ValueError("Unsupported structure type")
        

    def _serialize_blob(self, blob: Blob, subdivisions: int = 5) -> Trimesh:
        """
        Convert a blob structure to a triangular mesh.

        Parameters
        ----------
        blob : Blob
            The blob structure to serialize.
        subdivisions : int, optional
            Number of subdivisions for mesh quality. Default is 5.

        Returns
        -------
        Trimesh
            Triangular mesh with Perlin noise deformation applied.
        """
        unit_sphere = trimesh.primitives.Sphere(1, subdivisions=subdivisions)
        offsets = blob.calculate_offsets(unit_sphere.vertices)
        vertices = (1 + offsets) * unit_sphere.vertices
        mesh = trimesh.Trimesh(vertices=vertices, faces=unit_sphere.faces)
        mesh.apply_scale(blob.radius)
        mesh.apply_translation(blob.position)
        return mesh
    
    def _serialize_tube(self, tube: Tube, subdivisions: int = 5) -> Trimesh:
        """
        Convert a tube structure to a triangular mesh.

        Parameters
        ----------
        tube : Tube
            The tube structure to serialize.
        subdivisions : int, optional
            Number of subdivisions for mesh quality. Default is 5.

        Returns
        -------
        Trimesh
            Triangular mesh representing the cylindrical tube.
        """
        transform = (
            trimesh.transformations.translation_matrix(tube.position)
            @ trimesh.geometry.align_vectors([0, 0, 1], tube.direction)
        )
        return trimesh.creation.cylinder(
            radius=tube.radius,
            height=tube.height,
            sections=subdivisions ** 2,
            transform=transform
        )
