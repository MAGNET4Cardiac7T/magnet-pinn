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
from typing import Union, List
from abc import ABC, abstractmethod

import trimesh
import numpy as np
from trimesh import Trimesh

from .serializers import MeshSerializer
from .typing import MeshPhantom, StructurePhantom

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
        """
        Apply the transformation to input data.
        
        This method defines the core transformation logic that must be implemented
        by all concrete transform subclasses. It should process the input data
        and return the transformed result, maintaining the composable interface
        for pipeline construction.

        Parameters
        ----------
        *args
            Variable positional arguments passed to the transformation.
        **kwds
            Variable keyword arguments passed to the transformation.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement `__call__` method")
    
    def __repr__(self):
        """
        Return string representation of the transform.

        Returns
        -------
        str
            Class name formatted as a string for debugging and logging.
        """
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    Composite transform for chaining multiple transformation operations.
    
    Applies a sequence of transforms in order, passing the output of each
    transform as input to the next, enabling complex processing pipelines
    to be built from simple components.
    """

    def __init__(self, transforms: list[Transform], *args, **kwargs):
        """
        Initialize composite transform with a sequence of transforms.
        
        Creates a pipeline that applies each transform in the specified order,
        passing the output of each transform as input to the next. This enables
        the construction of complex processing workflows from simple, reusable
        transformation components.

        Parameters
        ----------
        transforms : list[Transform]
            Ordered list of transforms to apply sequentially. Each transform
            must implement the Transform interface with a callable method.
        *args, **kwargs
            Additional arguments passed to the parent Transform class.
        """
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __call__(self, original_phantom: PhantomType, *args, **kwargs):
        """
        Apply all transforms in sequence to the input phantom.
        
        Executes each transform in the pipeline order, passing the output of
        each transform as input to the next. This creates a processing chain
        where complex operations can be built from simple components.

        Parameters
        ----------
        original_phantom : PhantomType
            The input phantom to transform (StructurePhantom or MeshPhantom).
        *args, **kwds
            Additional arguments passed to each transform in the pipeline.

        Returns
        -------
        PhantomType
            The final transformed phantom after applying all pipeline transforms.
        """
        target_phantom = original_phantom
        for transform in self.transforms:
            target_phantom = transform(target_phantom, original_phantom, *args, **kwargs)
        return target_phantom

    def __repr__(self):
        """
        Return string representation of the composite transform.

        Returns
        -------
        str
            String showing the class name and ordered list of component transforms.
        """
        return f"{self.__class__.__name__}({', '.join([str(t) for t in self.transforms])})"


class ToMesh:
    """
    Transform for converting structure phantoms to mesh phantoms.
    
    Serializes abstract geometric structures (blobs, tubes) into concrete
    triangular mesh representations using the configured mesh serializer,
    preparing phantoms for geometric processing operations.
    """
    def __init__(self):
        """
        Initialize the structure-to-mesh converter.
        
        Sets up the mesh serializer that will be used to convert abstract
        geometric structures into concrete triangular mesh representations.
        The serializer handles different structure types and applies appropriate
        subdivision levels for mesh quality.

        Parameters
        ----------
        *args, **kwargs
            Additional arguments passed to the parent Transform class.
        """
        self.serializer = MeshSerializer()

    def __call__(self, tissue: StructurePhantom) -> MeshPhantom:
        """
        Convert a structure phantom to a mesh phantom.
        
        Transforms all geometric structures (parent blob, child blobs, tubes)
        into triangular mesh representations using the configured mesh serializer.
        This conversion prepares the phantom for subsequent geometric processing
        operations such as boolean cutting and mesh refinement.

        Parameters
        ----------
        tissue : StructurePhantom
            The structure phantom containing abstract geometric objects to convert.
        *args, **kwds
            Additional arguments (currently unused but maintained for interface consistency).

        Returns
        -------
        MeshPhantom
            Mesh phantom with triangular mesh representations of all components.
        """
        return MeshPhantom(
            parent=self.serializer.serialize(tissue.parent),
            children=[self.serializer.serialize(c) for c in tissue.children],
            tubes=[self.serializer.serialize(t) for t in tissue.tubes]
        )


class MeshesTubesClipping(Transform):
    def __call__(self, target_phantom: MeshPhantom, original_phantom: MeshPhantom, *args, **kwds) -> MeshPhantom:
        try:
            logging.debug("Attempting tubes clipping with manifold engine")

            _validate_input_meshes([original_phantom.parent] + original_phantom.tubes, "tubes clipping")
            _validate_input_meshes([target_phantom.parent] + target_phantom.tubes, "tubes clipping")

            if len(target_phantom.tubes) == 0:
                return target_phantom

            clipped_tubes = []
            for i, tube in enumerate(original_phantom.tubes):
                clipped_tube = trimesh.boolean.intersection([tube, original_phantom.parent], engine='manifold')
                clipped_tube.remove_degenerate_faces()
                # collapse duplicate or nearly‑duplicate faces/edges
                clipped_tube.remove_duplicate_faces()
                clipped_tube.remove_unreferenced_vertices()
                # stitch small holes
                clipped_tube.fill_holes()
                # make normals consistent
                trimesh.repair.fix_normals(clipped_tube)
                _validate_mesh(clipped_tube, f"tube {i} clipping")
                clipped_tubes.append(clipped_tube)
            
            logging.info("Tubes clipping successful")
            return MeshPhantom(
                parent=target_phantom.parent,
                children=target_phantom.children,
                tubes=clipped_tubes
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for tubes clipping. "
                f"Tubes count: {len(target_phantom.tubes)}, "
                f"Parent vertices: {len(original_phantom.parent.vertices)}, "
                f"Parent faces: {len(original_phantom.parent.faces)}, "
                f"Parent volume: {original_phantom.parent.volume:.3f}, "
                f"Error: {e}"
            )
        

class MeshesChildrenCutout(Transform):
    def __call__(self, target_phantom: MeshPhantom, original_phantom: MeshPhantom, *args, **kwds):
        try:
            logging.debug("Attempting children cutout with manifold engine")

            _validate_input_meshes(original_phantom.children + original_phantom.tubes, "children cutout")
            _validate_input_meshes(target_phantom.children + target_phantom.tubes, "children cutout")

            if len(target_phantom.tubes) == 0 or len(target_phantom.children) == 0:
                return target_phantom

            cutouts = []
            for i, child in enumerate(target_phantom.children):
                cutout = trimesh.boolean.difference([child, *original_phantom.tubes], engine='manifold')
                cutout.remove_degenerate_faces()
                # collapse duplicate or nearly‑duplicate faces/edges
                cutout.remove_duplicate_faces()
                cutout.remove_unreferenced_vertices()
                # stitch small holes
                cutout.fill_holes()
                # make normals consistent
                trimesh.repair.fix_normals(cutout)
                _validate_mesh(cutout, f"child {i} cutout")
                cutouts.append(cutout)

            logging.info("Children cutout successful")
            return MeshPhantom(
                parent=target_phantom.parent,
                children=cutouts,
                tubes=target_phantom.tubes
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for children cutout. "
                f"Children count: {len(target_phantom.children)}, "
                f"Parent vertices: {len(original_phantom.parent.vertices)}, "
                f"Parent faces: {len(original_phantom.parent.faces)}, "
                f"Parent volume: {original_phantom.parent.volume:.3f}, "
                f"Error: {e}"
            )
        

class MeshesParentCutoutWithChildren(Transform):
    def __call__(self, target_phantom: MeshPhantom, original_phantom: MeshPhantom, *args, **kwargs) -> MeshPhantom:
        try:
            logging.debug("Attempting parent cutout with children using manifold engine")

            _validate_input_meshes([original_phantom.parent] + original_phantom.children + original_phantom.tubes, "parent cutout with children")
            _validate_input_meshes([target_phantom.parent] + target_phantom.children + target_phantom.tubes, "parent cutout with children")

            if len(target_phantom.children) == 0:
                return target_phantom

            parent = trimesh.boolean.difference(
                [target_phantom.parent, *original_phantom.children],
                engine='manifold'
            )

            parent.remove_degenerate_faces()
            # collapse duplicate or nearly‑duplicate faces/edges
            parent.remove_duplicate_faces()
            parent.remove_unreferenced_vertices()
            # stitch small holes
            parent.fill_holes()
            # make normals consistent
            trimesh.repair.fix_normals(parent)

            _validate_mesh(parent, f"parent cutout with children result")

            return MeshPhantom(
                parent=parent,
                children=target_phantom.children,
                tubes=target_phantom.tubes
            )
        
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for parent cutout with children. "
                f"Parent vertices: {len(target_phantom.parent.vertices)}, "
                f"Parent faces: {len(target_phantom.parent.faces)}, "
                f"Parent volume: {target_phantom.parent.volume:.3f}, "
                f"Children count: {len(original_phantom.children)}, "
                f"Tubes count: {len(original_phantom.tubes)}, "
                f"Error: {e}"
            )
        

class MeshesParentCutoutWithTubes(Transform):
    def __call__(self, target_phantom: MeshPhantom, original_phantom: MeshPhantom, *args, **kwargs) -> MeshPhantom:
        try:
            logging.debug("Attempting parent cutout with tubes using manifold engine")

            _validate_input_meshes([original_phantom.parent] + original_phantom.tubes, "parent cutout with tubes")
            _validate_input_meshes([target_phantom.parent] + target_phantom.tubes, "parent cutout with tubes")

            if len(target_phantom.tubes) == 0:
                return target_phantom

            parent = trimesh.boolean.difference(
                [target_phantom.parent, *original_phantom.tubes],
                engine='manifold'
            )

            parent.remove_degenerate_faces()
            # collapse duplicate or nearly‑duplicate faces/edges
            parent.remove_duplicate_faces()
            parent.remove_unreferenced_vertices()
            # stitch small holes
            parent.fill_holes()
            # make normals consistent
            trimesh.repair.fix_normals(parent)

            _validate_mesh(parent, f"parent cutout with tubes result")

            return MeshPhantom(
                parent=parent,
                children=target_phantom.children,
                tubes=target_phantom.tubes
            )
        
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for parent cutout with tubes. "
                f"Parent vertices: {len(target_phantom.parent.vertices)}, "
                f"Parent faces: {len(target_phantom.parent.faces)}, "
                f"Parent volume: {target_phantom.parent.volume:.3f}, "
                f"Tubes count: {len(original_phantom.tubes)}, "
                f"Error: {e}"
            )
    

class MeshesChildrenClipping(Transform):
    def __call__(self, target_phantom: MeshPhantom, original_phantom: MeshPhantom, *args, **kwargs) -> MeshPhantom:
        try:
            logging.debug("Attempting children clipping with manifold engine")

            _validate_input_meshes([original_phantom.parent] + original_phantom.children, "children clipping")
            _validate_input_meshes([target_phantom.parent] + target_phantom.children, "children clipping")

            if len(target_phantom.children) == 0:
                return target_phantom

            clipped_children = []
            for i, child in enumerate(target_phantom.children):
                clipped_child = trimesh.boolean.intersection([child, original_phantom.parent], engine='manifold')
                clipped_child.remove_degenerate_faces()
                # collapse duplicate or nearly‑duplicate faces/edges
                clipped_child.remove_duplicate_faces()
                clipped_child.remove_unreferenced_vertices()
                # stitch small holes
                clipped_child.fill_holes()
                # make normals consistent
                trimesh.repair.fix_normals(clipped_child)
                _validate_mesh(clipped_child, f"child {i} clipping")
                clipped_children.append(clipped_child)
            
            logging.info("Children clipping successful")
            return MeshPhantom(
                parent=target_phantom.parent,
                children=clipped_children,
                tubes=target_phantom.tubes
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for children clipping. "
                f"Children count: {len(target_phantom.children)}, "
                f"Parent vertices: {len(original_phantom.parent.vertices)}, "
                f"Parent faces: {len(original_phantom.parent.faces)}, "
                f"Parent volume: {original_phantom.parent.volume:.3f}, "
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
        """
        Clean and repair all mesh components in the phantom.
        
        Applies comprehensive mesh cleaning operations to all phantom components
        to ensure high-quality geometry suitable for downstream processing. The
        cleaning process removes degenerate faces, fills holes, merges duplicate
        vertices, fixes normal orientations, and removes unreferenced vertices
        to create robust mesh representations.

        Parameters
        ----------
        tissue : MeshPhantom
            The mesh phantom containing components to clean and repair.
        *args, **kwds
            Additional arguments (currently unused but maintained for interface consistency).

        Returns
        -------
        MeshPhantom
            Cleaned mesh phantom with improved geometry quality for all components.
        """
        return MeshPhantom(
            parent=self._clean_mesh(tissue.parent),
            children=[self._clean_mesh(c) for c in tissue.children],
            tubes=[self._clean_mesh(t) for t in tissue.tubes]
        )
    
    def _clean_mesh(self, mesh: Trimesh) -> Trimesh:
        """
        Apply comprehensive cleaning operations to a single mesh.
        
        Performs a sequence of mesh repair operations including degenerate face
        removal, duplicate face elimination, hole filling, vertex merging, normal
        fixing, and unreferenced vertex removal. These operations ensure the mesh
        is suitable for boolean operations and simulation workflows.

        Parameters
        ----------
        mesh : Trimesh
            The mesh to clean and repair.

        Returns
        -------
        Trimesh
            Cleaned mesh with improved geometric quality and consistency.
        """
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
        """
        Initialize the adaptive mesh refinement transform.
        
        Sets up the remeshing parameters for subdividing mesh elements to achieve
        uniform edge lengths. The maximum edge length threshold controls the level
        of mesh refinement and affects the trade-off between geometric accuracy
        and computational complexity.

        Parameters
        ----------
        max_len : float, optional
            Maximum edge length threshold for remeshing. Default is 8.0.
            Smaller values create finer meshes with more elements.
        *args, **kwargs
            Additional arguments passed to the parent Transform class.
        """
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        """
        Apply adaptive mesh refinement to all phantom components.
        
        Subdivides mesh elements in all phantom components to ensure edge lengths
        are below the specified threshold. This creates more uniform mesh quality
        suitable for numerical simulations while maintaining geometric fidelity.
        The operation may result in non-watertight meshes due to subdivision
        algorithm limitations.

        Parameters
        ----------
        tissue : MeshPhantom
            The mesh phantom containing components to remesh.
        *args, **kwds
            Additional arguments (currently unused but maintained for interface consistency).

        Returns
        -------
        MeshPhantom
            Remeshed phantom with refined mesh quality for all components.
        """
        return MeshPhantom(
            parent=self._remesh(tissue.parent),
            children=[self._remesh(c) for c in tissue.children],
            tubes=[self._remesh(t) for t in tissue.tubes]
        )
    
    def _remesh(self, mesh: Trimesh) -> Trimesh:
        """
        Apply subdivision remeshing to achieve uniform edge lengths.
        
        Subdivides mesh elements iteratively until all edges are below the
        maximum length threshold. This creates more uniform element sizes
        for improved numerical accuracy in simulations, though it may increase
        mesh complexity significantly.

        Parameters
        ----------
        mesh : Trimesh
            The mesh to subdivide and refine.

        Returns
        -------
        Trimesh
            Remeshed geometry with edges below the maximum length threshold.
        """
        v, f = trimesh.remesh.subdivide_to_size(
            mesh.vertices, 
            mesh.faces, 
            max_edge=self.max_len
        )
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        return mesh
