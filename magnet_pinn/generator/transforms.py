"""
NAME
    transforms.py

DESCRIPTION
    This module provides transformation pipeline components for phantom processing.
    Contains composable transforms for converting phantoms between different representations
    (structures to meshes) and applying geometric operations like boolean mesh cutting,
    cleaning, and remeshing to produce simulation-ready outputs.
"""
import logging
from typing import Union
from abc import ABC, abstractmethod

import trimesh
from trimesh import Trimesh

from .serializers import MeshSerializer
from .typing import MeshPhantom, StructurePhantom

# Type alias for backward compatibility
PhantomType = Union[StructurePhantom, MeshPhantom]


def _validate_mesh(mesh: Trimesh, operation_name: str) -> None:
    """Validate mesh quality after boolean operations."""
    if mesh is None:
        raise ValueError(f"Mesh is None after {operation_name}")
    
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices after {operation_name}")
    
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces after {operation_name}")
    
    if not mesh.is_volume:
        logging.warning(f"Mesh is not a valid volume after {operation_name}")
    
    if mesh.volume <= 0:
        raise ValueError(f"Mesh has invalid volume {mesh.volume} after {operation_name}")


def _validate_input_meshes(meshes: list[Trimesh], operation_name: str) -> None:
    """Validate input meshes before boolean operations."""
    for i, mesh in enumerate(meshes):
        if mesh is None:
            raise ValueError(f"Input mesh {i} is None for {operation_name}")
        if len(mesh.vertices) == 0:
            raise ValueError(f"Input mesh {i} has no vertices for {operation_name}")
        if len(mesh.faces) == 0:
            raise ValueError(f"Input mesh {i} has no faces for {operation_name}")


class Transform(ABC):
    """
    Abstract base class for phantom transformation operations.
    
    Provides the interface for composable transformation components that can
    be chained together to build complex phantom processing pipelines.
    """

    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError("Subclasses must implement `__call__` method")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    Composite transform for chaining multiple transformation operations.
    
    Applies a sequence of transforms in order, passing the output of each
    transform as input to the next, enabling complex processing pipelines
    to be built from simple components.
    """

    def __init__(self, transforms: list[Transform], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __call__(self, tissue: PhantomType, *args, **kwds) :
        for transform in self.transforms:
            tissue = transform(tissue, *args, **kwds)
        return tissue

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([str(t) for t in self.transforms])})"
    

class ToMesh(Transform):
    """
    Transform for converting structure phantoms to mesh phantoms.
    
    Serializes abstract geometric structures (blobs, tubes) into concrete
    triangular mesh representations using the configured mesh serializer,
    preparing phantoms for geometric processing operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serializer = MeshSerializer()

    def __call__(self, tissue: StructurePhantom, *args, **kwds) -> MeshPhantom:
        return MeshPhantom(
            parent=self.serializer.serialize(tissue.parent),
            children=[self.serializer.serialize(c) for c in tissue.children],
            tubes=[self.serializer.serialize(t) for t in tissue.tubes]
        )


class MeshesCutout(Transform):
    """
    Transform for applying boolean cutting operations between phantom components.
    
    Performs boolean subtraction to cut child blobs and tubes out of the parent
    blob and tubes out of child blobs, creating realistic anatomical cavities
    and ensuring proper geometric relationships between components.
    """

    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        try:
            return MeshPhantom(
                parent=self._cut_parent(tissue),
                children=self._cut_children(tissue),
                tubes=self._cut_tubes(tissue)
            )
        except (RuntimeError, ValueError) as e:
            logging.error(f"MeshesCutout failed: {e}")
            # Re-raise to allow calling code to handle appropriately
            raise

    def _cut_parent(self, tissue: MeshPhantom) -> Trimesh:
        cutters = tissue.children + tissue.tubes
        if not cutters:
            return tissue.parent
        
        try:
            logging.debug("Attempting parent cutting with manifold engine")
            
            # Validate input meshes
            _validate_input_meshes([tissue.parent] + cutters, "parent cutting")
            
            # Create union of all cutters
            union_cutters = trimesh.boolean.union(cutters, engine='manifold')
            _validate_mesh(union_cutters, "union operation")
            
            # Cut parent with union of cutters
            result = trimesh.boolean.difference([tissue.parent, union_cutters], engine='manifold')
            _validate_mesh(result, "parent difference operation")
            
            logging.info("Parent cutting successful")
            return result
            
        except Exception as e:
            raise RuntimeError(
                f"Boolean operation failed for parent cutting. "
                f"Parent vertices: {len(tissue.parent.vertices)}, "
                f"Parent faces: {len(tissue.parent.faces)}, "
                f"Parent volume: {tissue.parent.volume:.3f}, "
                f"Cutters count: {len(cutters)}, "
                f"Error: {e}"
            )

    def _cut_children(self, tissue: MeshPhantom) -> list[Trimesh]:
        if not tissue.tubes:
            return tissue.children
        
        try:
            logging.debug("Attempting children cutting with manifold engine")
            
            # Validate input meshes
            _validate_input_meshes(tissue.children + tissue.tubes, "children cutting")
            
            # Create union of all tubes
            tubes_union = trimesh.boolean.union(tissue.tubes, engine='manifold')
            _validate_mesh(tubes_union, "tubes union")
            
            # Cut each child with tubes union
            result_children = []
            for i, child in enumerate(tissue.children):
                try:
                    cut_child = trimesh.boolean.difference([child, tubes_union], engine='manifold')
                    _validate_mesh(cut_child, f"child {i} cutting")
                    result_children.append(cut_child)
                except Exception as child_error:
                    raise RuntimeError(
                        f"Failed to cut child {i} (vertices: {len(child.vertices)}, "
                        f"faces: {len(child.faces)}, volume: {child.volume:.3f}): {child_error}"
                    )
            
            logging.info("Children cutting successful")
            return result_children
            
        except RuntimeError:
            # Re-raise our custom exception
            raise
        except Exception as e:
            raise RuntimeError(
                f"Boolean operation failed for children cutting. "
                f"Children count: {len(tissue.children)}, "
                f"Tubes count: {len(tissue.tubes)}, "
                f"Error: {e}"
            )
    
    def _cut_tubes(self, tissue: MeshPhantom) -> list[Trimesh]:
        try:
            logging.debug("Attempting tube cutting with manifold engine")
            
            # Validate input meshes
            _validate_input_meshes([tissue.parent] + tissue.tubes, "tube cutting")
            
            # Intersect each tube with parent
            result_tubes = []
            for i, tube in enumerate(tissue.tubes):
                try:
                    cut_tube = trimesh.boolean.intersection([tube, tissue.parent], engine='manifold')
                    _validate_mesh(cut_tube, f"tube {i} intersection")
                    result_tubes.append(cut_tube)
                except Exception as tube_error:
                    raise RuntimeError(
                        f"Failed to cut tube {i} (vertices: {len(tube.vertices)}, "
                        f"faces: {len(tube.faces)}, volume: {tube.volume:.3f}): {tube_error}"
                    )
            
            logging.info("Tube cutting successful")
            return result_tubes
            
        except RuntimeError:
            # Re-raise our custom exception
            raise
        except Exception as e:
            raise RuntimeError(
                f"Boolean operation failed for tube cutting. "
                f"Tubes count: {len(tissue.tubes)}, "
                f"Parent vertices: {len(tissue.parent.vertices)}, "
                f"Parent faces: {len(tissue.parent.faces)}, "
                f"Parent volume: {tissue.parent.volume:.3f}, "
                f"Error: {e}"
            )
    

class MeshesCleaning(Transform):
    """
    Transform for cleaning and repairing mesh geometry after boolean operations.
    
    Applies various mesh repair operations including degenerate face removal,
    hole filling, vertex merging, normal fixing, and unreferenced vertex removal
    to ensure mesh quality for subsequent processing.
    """
    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        return MeshPhantom(
            parent=self._clean_mesh(tissue.parent),
            children=[self._clean_mesh(c) for c in tissue.children],
            tubes=[self._clean_mesh(t) for t in tissue.tubes]
        )
    
    def _clean_mesh(self, mesh: Trimesh) -> Trimesh:
        mesh = mesh.copy()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        trimesh.repair.fill_holes(mesh)
        mesh.merge_vertices()
        mesh.fix_normals()
        mesh.remove_unreferenced_vertices()
        return mesh
    

class MeshesRemesh(Transform):
    """
    Transform for adaptive mesh refinement and subdivision.
    
    Subdivides mesh elements to achieve uniform edge lengths below a specified
    threshold, improving mesh quality for numerical simulations. Note that this
    operation may produce non-watertight meshes due to the underlying subdivision
    algorithm limitations.
    """

    def __init__(self, max_len: float = 8.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        return MeshPhantom(
            parent=self._remesh(tissue.parent),
            children=[self._remesh(c) for c in tissue.children],
            tubes=[self._remesh(t) for t in tissue.tubes]
        )
    
    def _remesh(self, mesh: Trimesh) -> Trimesh:
        v, f = trimesh.remesh.subdivide_to_size(
            mesh.vertices, 
            mesh.faces, 
            max_edge=self.max_len
        )
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        return mesh
