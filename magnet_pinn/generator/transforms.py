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

    def __call__(self, tissue: PhantomType, *args, **kwargs):
        """
        Apply all transforms in sequence to the input phantom.
        
        Executes each transform in the pipeline order, passing the output of
        each transform as input to the next. This creates a processing chain
        where complex operations can be built from simple components.

        Parameters
        ----------
        tissue : PhantomType
            The input phantom to transform (StructurePhantom or MeshPhantom).
        *args, **kwds
            Additional arguments passed to each transform in the pipeline.

        Returns
        -------
        PhantomType
            The final transformed phantom after applying all pipeline transforms.
        """
        for transform in self.transforms:
            tissue = transform(tissue, *args, **kwargs)
        return tissue

    def __repr__(self):
        """
        Return string representation of the composite transform.

        Returns
        -------
        str
            String showing the class name and ordered list of component transforms.
        """
        return f"{self.__class__.__name__}({', '.join([str(t) for t in self.transforms])})"


class MeshesApplyInParallel(Transform):
    def __init__(self, transforms: list[Transform], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __call__(self, tissue: MeshPhantom, *args, **kwargs) -> MeshPhantom:
        results = []
        for transform in self.transforms:
            transform_result = transform(tissue, *args, **kwargs)
            results.append(transform_result)

        return self._merge_results(results)

    def _merge_results(self, results) -> MeshPhantom:
        parents = []
        children = []
        tubes = []
        for result in results:
            if result.parent is not None:
                parents.append(result.parent)

            if result.children is not None:
                subchildren_list = [
                    child for child in result.children
                ]
                children.append(subchildren_list)
            if result.tubes is not None:
                subtubes_list = [
                    tube for tube in result.tubes
                ]
                tubes.append(subtubes_list)

        children = np.array(children)
        children = [
            trimesh.boolean.intersection(children[:, i].tolist(), engine='manifold')
            for i in range(children.shape[1])
        ]

        tubes = np.array(tubes)
        tubes = [
            trimesh.boolean.intersection(tubes[:, i].tolist(), engine='manifold')
            for i in range(tubes.shape[1])
        ]

        return MeshPhantom(
            parent=trimesh.boolean.intersection(parents, engine='manifold'),
            children=children,
            tubes=tubes
        )

class MeshProc(Transform):

    def __call__(self, original_tissue, processed_tissue, *args, **kwds):
        return super().__call__(*args, **kwds)

    

class ToMesh(Transform):
    """
    Transform for converting structure phantoms to mesh phantoms.
    
    Serializes abstract geometric structures (blobs, tubes) into concrete
    triangular mesh representations using the configured mesh serializer,
    preparing phantoms for geometric processing operations.
    """
    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.serializer = MeshSerializer()

    def __call__(self, tissue: StructurePhantom, *args, **kwds) -> MeshPhantom:
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
    

class MeshesClipping(Transform):
    """
    Transform for clipping mesh components to stay within parental boundaries.
    
    Ensures that all child blobs and tubes remain within the parental mesh
    boundaries by performing boolean intersection operations. This prevents
    child components from extending outside the anatomical container and
    maintains realistic geometric relationships.
    """

    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        """
        Clip all mesh components to stay within parental boundaries.
        
        Performs boolean intersection operations to ensure that all child
        blobs and tubes remain within the confines of the parent mesh.
        This maintains anatomical realism by preventing components from
        extending beyond their container boundaries.

        Parameters
        ----------
        tissue : MeshPhantom
            The mesh phantom containing parent, children, and tube meshes to process.
        *args, **kwds
            Additional arguments (currently unused but maintained for interface consistency).

        Returns
        -------
        MeshPhantom
            Processed mesh phantom with all components clipped to parent boundaries.

        Raises
        ------
        RuntimeError
            If boolean operations fail due to invalid mesh geometry or engine errors.
        ValueError
            If input meshes are invalid or have zero volume.
        """
        try:
            return MeshPhantom(
                parent=tissue.parent,  # Parent remains unchanged
                children=self._clip_children(tissue),
                tubes=self._clip_tubes(tissue)
            )
        except (RuntimeError, ValueError) as e:
            logging.error(f"MeshesClipping failed: {e}")
            raise

    def _clip_children(self, tissue: MeshPhantom) -> list[Trimesh]:
        """
        Clip all child meshes to stay within parent boundaries.
        
        Performs boolean intersection between each child mesh and the parent
        mesh to ensure children remain within anatomical boundaries. Any
        portions of child meshes extending outside the parent are removed.

        Parameters
        ----------
        tissue : MeshPhantom
            The mesh phantom containing meshes to process.

        Returns
        -------
        list[Trimesh]
            List of child meshes clipped to parent boundaries.

        Raises
        ------
        RuntimeError
            If intersection operations fail for any child mesh.
        """
        if not tissue.children:
            return tissue.children
        
        try:
            logging.debug("Attempting children clipping with manifold engine")
            
            _validate_input_meshes([tissue.parent] + tissue.children, "children clipping")
            
            clipped_children = []
            for i, child in enumerate(tissue.children):
                try:
                    clipped_child = trimesh.boolean.intersection([child, tissue.parent], engine='manifold')
                    _validate_mesh(clipped_child, f"child {i} clipping")
                    clipped_children.append(clipped_child)
                except Exception as child_error:
                    raise RuntimeError(
                        f"Failed to clip child {i} (vertices: {len(child.vertices)}, "
                        f"faces: {len(child.faces)}, volume: {child.volume:.3f}): {child_error}"
                    )
            
            logging.info("Children clipping successful")
            return clipped_children
            
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Boolean operation failed for children clipping. "
                f"Children count: {len(tissue.children)}, "
                f"Parent vertices: {len(tissue.parent.vertices)}, "
                f"Parent faces: {len(tissue.parent.faces)}, "
                f"Parent volume: {tissue.parent.volume:.3f}, "
                f"Error: {e}"
            )

    def _clip_tubes(self, tissue: MeshPhantom) -> list[Trimesh]:
        """
        Clip all tube meshes to stay within parent boundaries.
        
        Performs boolean intersection between each tube mesh and the parent
        mesh to ensure tubes remain within anatomical boundaries. Any
        portions of tube meshes extending outside the parent are removed.

        Parameters
        ----------
        tissue : MeshPhantom
            The mesh phantom containing meshes to process.

        Returns
        -------
        list[Trimesh]
            List of tube meshes clipped to parent boundaries.

        Raises
        ------
        RuntimeError
            If intersection operations fail for any tube mesh.
        """
        if not tissue.tubes:
            return tissue.tubes
        
        try:
            logging.debug("Attempting tubes clipping with manifold engine")
            
            _validate_input_meshes([tissue.parent] + tissue.tubes, "tubes clipping")
            
            clipped_tubes = []
            for i, tube in enumerate(tissue.tubes):
                try:
                    clipped_tube = trimesh.boolean.intersection([tube, tissue.parent], engine='manifold')
                    _validate_mesh(clipped_tube, f"tube {i} clipping")
                    clipped_tubes.append(clipped_tube)
                except Exception as tube_error:
                    raise RuntimeError(
                        f"Failed to clip tube {i} (vertices: {len(tube.vertices)}, "
                        f"faces: {len(tube.faces)}, volume: {tube.volume:.3f}): {tube_error}"
                    )
            
            logging.info("Tubes clipping successful")
            return clipped_tubes
            
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Boolean operation failed for tubes clipping. "
                f"Tubes count: {len(tissue.tubes)}, "
                f"Parent vertices: {len(tissue.parent.vertices)}, "
                f"Parent faces: {len(tissue.parent.faces)}, "
                f"Parent volume: {tissue.parent.volume:.3f}, "
                f"Error: {e}"
            )
        

class MeshesChildrenClipping(Transform):
    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        try:
            logging.debug("Attempting children clipping with manifold engine")
            
            _validate_input_meshes([tissue.parent] + tissue.children, "children clipping")
            
            clipped_children = []
            for i, child in enumerate(tissue.children):
                clipped_child = trimesh.boolean.intersection([child, tissue.parent], engine='blender')
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
                parent=None,
                children=clipped_children,
                tubes=None
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for children clipping. "
                f"Children count: {len(tissue.children)}, "
                f"Parent vertices: {len(tissue.parent.vertices)}, "
                f"Parent faces: {len(tissue.parent.faces)}, "
                f"Parent volume: {tissue.parent.volume:.3f}, "
                f"Error: {e}"
            )
        

class MeshesTubesClipping(Transform):
    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        try:
            logging.debug("Attempting tubes clipping with manifold engine")
            
            _validate_input_meshes([tissue.parent] + tissue.tubes, "tubes clipping")
            
            clipped_tubes = []
            for i, tube in enumerate(tissue.tubes):
                clipped_tube = trimesh.boolean.intersection([tube, tissue.parent], engine='blender')
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
                parent=None,
                children=None,
                tubes=clipped_tubes
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for tubes clipping. "
                f"Tubes count: {len(tissue.tubes)}, "
                f"Parent vertices: {len(tissue.parent.vertices)}, "
                f"Parent faces: {len(tissue.parent.faces)}, "
                f"Parent volume: {tissue.parent.volume:.3f}, "
                f"Error: {e}"
            )
        

class MeshesChildrenCutout(Transform):
    def __call__(self, phantom: MeshPhantom, *args, **kwds):
        try:
            logging.debug("Attempting children cutout with manifold engine")

            _validate_input_meshes([phantom.parent] + phantom.children + phantom.tubes, "children cutout")

            cutouts = []
            for i, child in enumerate(phantom.children):
                cutout = trimesh.boolean.difference([child, *phantom.tubes], engine='blender')
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
                parent=None,
                children=cutouts,
                tubes=None
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for children cutout. "
                f"Children count: {len(phantom.children)}, "
                f"Parent vertices: {len(phantom.parent.vertices)}, "
                f"Parent faces: {len(phantom.parent.faces)}, "
                f"Parent volume: {phantom.parent.volume:.3f}, "
                f"Error: {e}"
            )

class MeshesParentCutout(Transform):

    def __call__(self, phantom: MeshPhantom, *args, **kwargs) -> MeshPhantom:
        try:
            logging.debug("Attempting parent cutout with manifold engine")

            _validate_input_meshes([phantom.parent] + phantom.children + phantom.tubes, "parent cutout")

            parent = trimesh.boolean.difference(
                [phantom.parent, *phantom.children, *phantom.tubes],
                engine='blender'
            )

            parent.remove_degenerate_faces()
            # collapse duplicate or nearly‑duplicate faces/edges
            parent.remove_duplicate_faces()
            parent.remove_unreferenced_vertices()
            # stitch small holes
            parent.fill_holes()
            # make normals consistent
            trimesh.repair.fix_normals(parent)

            _validate_mesh(parent, f"parent cutout result")

            return MeshPhantom(
                parent=parent,
                children=None,
                tubes=None
            )
        
        except RuntimeError as e:
            raise RuntimeError(
                f"Boolean operation failed for parent cutout. "
                f"Parent vertices: {len(phantom.parent.vertices)}, "
                f"Parent faces: {len(phantom.parent.faces)}, "
                f"Parent volume: {phantom.parent.volume:.3f}, "
                f"Children count: {len(phantom.children)}, "
                f"Tubes count: {len(phantom.tubes)}, "
                f"Error: {e}"
            )
        


class MeshesCutout(Transform):
    """
    Transform for applying boolean cutting operations between phantom components.
    
    Performs boolean subtraction to cut child blobs and tubes out of the parent
    blob and tubes out of child blobs, creating realistic anatomical cavities
    and ensuring proper geometric relationships between components.
    """

    def __call__(self, tissue: MeshPhantom, *args, **kwds) -> MeshPhantom:
        """
        Apply boolean cutting operations to create anatomical cavities.
        
        Performs comprehensive boolean subtraction operations to create realistic
        anatomical relationships between phantom components. Child blobs and tubes
        are cut out of the parent blob, and tubes are cut out of child blobs,
        resulting in proper geometric containment and void spaces that represent
        internal structures.

        Parameters
        ----------
        tissue : MeshPhantom
            The mesh phantom containing parent, children, and tube meshes to process.
        *args, **kwds
            Additional arguments (currently unused but maintained for interface consistency).

        Returns
        -------
        MeshPhantom
            Processed mesh phantom with boolean cutting operations applied.

        Raises
        ------
        RuntimeError
            If boolean operations fail due to invalid mesh geometry or engine errors.
        ValueError
            If input meshes are invalid or have zero volume.
        """
        try:
            return MeshPhantom(
                parent=self._cut_parent(tissue),
                children=self._cut_children(tissue),
                tubes=self._cut_tubes(tissue)
            )
        except (RuntimeError, ValueError) as e:
            logging.error(f"MeshesCutout failed: {e}")
            raise

    def _cut_parent(self, tissue: MeshPhantom) -> Trimesh:
        cutters = tissue.children + tissue.tubes
        if not cutters:
            return tissue.parent
        
        try:
            logging.debug("Attempting parent cutting with manifold engine")
            
            _validate_input_meshes([tissue.parent] + cutters, "parent cutting")
            
            union_cutters = trimesh.boolean.union(cutters, engine='manifold')
            _validate_mesh(union_cutters, "union operation")
            
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
            
            _validate_input_meshes(tissue.children + tissue.tubes, "children cutting")
            
            tubes_union = trimesh.boolean.union(tissue.tubes, engine='manifold')
            _validate_mesh(tubes_union, "tubes union")
            
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
            
            _validate_input_meshes([tissue.parent] + tissue.tubes, "tube cutting")
            
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
