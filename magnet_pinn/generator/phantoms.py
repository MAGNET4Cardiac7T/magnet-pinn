"""
NAME
    phantoms.py

DESCRIPTION
    This module provides high-level phantom generation classes for MRI simulation.
    Contains complex generative objects that combine multiple geometric structures
    to create realistic tissue phantoms with anatomical features like blood vessels
    and hierarchical blob structures. Configuration parameters for structure generation
    are passed to the appropriate sampler constructors during phantom initialization,
    while runtime parameters are provided during the generation process.
"""
import logging
from abc import ABC

import numpy as np
from numpy.random import default_rng

from .typing import StructurePhantom
from .samplers import (
    BlobSampler, TubeSampler, MeshBlobSampler, MeshTubeSampler
)
from .structures import Blob, CustomMeshStructure


class Phantom(ABC):
    """
    Abstract base class for phantom generators.
    
    Defines the common interface for generating structured phantoms with
    configurable geometric parameters and reproducible random generation
    using seed values. All concrete phantom implementations must inherit
    from this class and implement the generate method.

    Attributes
    ----------
    initial_blob_radius : float
        Base radius for the primary blob structure in the phantom.
        Must be a positive value representing the characteristic size scale.
    initial_blob_center_extent : np.ndarray
        Array-like structure defining the spatial bounds for phantom placement.
        Contains coordinate ranges for initial blob center positioning.
        First X, then Y, then Z dimensions.
    """
    initial_blob_radius: float
    initial_blob_center_extent: np.ndarray

    def __init__(self, initial_blob_radius: float, initial_blob_center_extent: np.ndarray):
        """
        Initialize phantom generator with basic geometric parameters.

        Parameters
        ----------
        initial_blob_radius : float
            Base radius for the primary blob structure. Must be positive.
        initial_blob_center_extent : np.ndarray
            Spatial bounds array for phantom center placement.
            First X, then Y, then Z dimensions.
        """
        self.initial_blob_radius = initial_blob_radius
        self.initial_blob_center_extent = initial_blob_center_extent

    def generate(self, seed: int = None) -> StructurePhantom:
        """
        Generate a complete phantom structure.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible generation. If None, uses system time.

        Returns
        -------
        StructurePhantom
            Complete phantom structure with all geometric components.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Tissue(Phantom):
    """
    Generator for complex tissue phantoms with hierarchical blob structures and vasculature.
    
    Creates realistic tissue phantoms by combining a parent blob with multiple child blobs
    and a network of tube structures representing blood vessels. The generator uses 
    sphere packing algorithms to ensure non-overlapping placement and provides extensive
    validation to prevent invalid configurations. Child blobs are scaled versions of the
    parent blob, positioned using progressive sampling for efficient collision avoidance.
    Tubes are generated to represent vascular structures with configurable radius ranges.

    Attributes
    ----------
    num_children_blobs : int
        Number of child blob structures to generate within the parent blob.
        Must be non-negative. Zero creates a phantom with only parent and tubes.
    num_tubes : int
        Number of tube structures to generate for vascular representation.
        Must be non-negative. Tubes are placed to avoid intersections.
    blob_sampler : BlobSampler
        Sampler instance configured with blob generation parameters.
    tube_sampler : TubeSampler
        Sampler instance configured with tube generation parameters.
    """
    num_children_blobs: int
    num_tubes: int
    blob_sampler: BlobSampler
    tube_sampler: TubeSampler
    
    def __init__(self, 
                 num_children_blobs: int, 
                 initial_blob_radius: float, 
                 initial_blob_center_extent: np.ndarray,
                 blob_radius_decrease_per_level: float, 
                 num_tubes: int, 
                 relative_tube_max_radius: float,
                 relative_tube_min_radius: float = 0.01):
        """
        Initialize tissue phantom generator with hierarchical structure parameters.
        
        Validates all input parameters to ensure geometrically feasible phantom
        configurations. Creates configured sampler instances for blob and tube
        generation. Sets up internal parameters for phantom generation.

        Parameters
        ----------
        num_children_blobs : int
            Number of child blobs to generate. Must be non-negative.
        initial_blob_radius : float
            Radius of the parent blob structure. Must be positive.
        initial_blob_center_extent : np.ndarray
            Array defining spatial bounds for parent blob center placement. 
            First X, then Y, then Z.
            Should contain coordinate ranges for each spatial dimension.
        blob_radius_decrease_per_level : float
            Scaling factor for child blob radii relative to parent radius.
            Must be in range (0, 1) to ensure children are smaller than parent.
        num_tubes : int
            Number of tube structures to generate. Must be non-negative.
        relative_tube_max_radius : float
            Maximum tube radius as fraction of parent blob radius.
            Must be in range (0, 1) to ensure tubes fit within parent.
        relative_tube_min_radius : float, optional
            Minimum tube radius as fraction of parent blob radius.
            Must be positive and less than max radius. Default is 0.01.

        Raises
        ------
        ValueError
            If any parameter is outside valid range or violates geometric constraints.
        """
        
        if num_children_blobs < 0:
            raise ValueError("num_children_blobs must be non-negative")
        if initial_blob_radius <= 0:
            raise ValueError("initial_blob_radius must be positive")
        if num_tubes < 0:
            raise ValueError("num_tubes must be non-negative")
        if relative_tube_max_radius <= 0 or relative_tube_max_radius >= 1:
            raise ValueError("relative_tube_max_radius must be in (0, 1)")
        if relative_tube_min_radius <= 0 or relative_tube_min_radius >= relative_tube_max_radius:
            raise ValueError("relative_tube_min_radius must be in (0, relative_tube_max_radius)")
        if initial_blob_center_extent is None or (isinstance(initial_blob_center_extent, list) and all(isinstance(elem, list) for elem in initial_blob_center_extent)):
            raise ValueError("initial_blob_center_extent must be a 2d array-like structure with coordinate ranges, first X, then Y, then Z dimensions")
        
        super().__init__(initial_blob_radius, initial_blob_center_extent)

        self.num_children_blobs = num_children_blobs
        self.num_tubes = num_tubes
        
        self.blob_sampler = BlobSampler(blob_radius_decrease_per_level)
        
        tube_max_radius = relative_tube_max_radius * initial_blob_radius
        tube_min_radius = relative_tube_min_radius * initial_blob_radius
        self.tube_sampler = TubeSampler(tube_max_radius, tube_min_radius, initial_blob_radius)

    def generate(self, seed: int = None) -> StructurePhantom:
        """
        Generate a complete tissue phantom with hierarchical structure and vasculature.
        
        Creates a realistic tissue phantom by generating a parent blob, positioning
        child blobs within it using progressive sampling, and adding tube structures
        for vascular representation. All sampling operations use the provided seed
        for reproducible results. The generation process includes extensive validation
        and collision detection to ensure geometrically valid configurations.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible phantom generation. If None, uses
            system-generated random seed for non-deterministic results.

        Returns
        -------
        StructurePhantom
            Complete phantom structure containing:
            - parent: Main Blob structure defining the tissue boundary
            - children: List of child Blob structures positioned within parent
            - tubes: List of Tube structures representing vascular network

        Raises
        ------
        RuntimeError
            If geometric constraints cannot be satisfied or sphere packing fails.
        """
        gen_object = default_rng(seed)

        initial_blob_center = np.array([
            gen_object.uniform(dim[0], dim[1])
            for dim in self.initial_blob_center_extent
        ])

        logging.info("Generating parent blob.")
        parent_blob = Blob(initial_blob_center, self.initial_blob_radius, seed=gen_object.integers(0, 2**32-1).item())

        logging.info("Generating children blobs.")
        children_blobs = self.blob_sampler.sample_children_blobs(
            parent_blob=parent_blob,
            num_children=self.num_children_blobs,
            rng=gen_object
        )

        logging.info("Generating tubes.")
        parent_inner_radius = parent_blob.radius*(1+parent_blob.empirical_min_offset)
        tubes = self.tube_sampler.sample_tubes(
            center=initial_blob_center, 
            radius=parent_inner_radius, 
            num_tubes=self.num_tubes,
            rng=gen_object
        )

        return StructurePhantom(
            parent=parent_blob,
            children=children_blobs,
            tubes=tubes
        )


class CustomPhantom(Phantom):
    """
    Generator for custom phantoms based on STL mesh structures.

    Creates realistic tissue phantoms by combining a parental mesh with multiple child blobs
    and a network of tube structures representing blood vessels. The generator uses sampling 
    points inside mesh to ensure children blobs are placed correctly inside the parental mesh.
    Tubes are generated with configurable radius ranges.

        Attributes
    ----------
    parent_structure : CustomMeshStructure
        The parent mesh structure loaded from the provided STL file path.
    num_children_blobs : int
        Number of child blob structures to generate within the parent blob.
        Must be non-negative. Zero creates a phantom with only parent and tubes.
    num_tubes : int
        Number of tube structures to generate for vascular representation.
        Must be non-negative. Tubes are placed to avoid intersections.
    blob_sampler : MeshBlobSampler
        Sampler instance configured with blob generation parameters.
    tube_sampler : MeshTubeSampler
        Sampler instance configured with tube generation parameters.
    sample_children_only_inside : bool
        If True, child blobs are sampled fully inside the parent mesh
    """
    def __init__(self, stl_mesh_path: str, num_children_blobs: int = 3, 
                 blob_radius_decrease_per_level: float = 0.3, num_tubes: int = 5,
                 relative_tube_max_radius: float = 0.1, relative_tube_min_radius: float = 0.01,
                 sample_children_only_inside: bool = False):
        self.parent_structure = CustomMeshStructure(stl_mesh_path)

        super().__init__(None, None)

        child_radius = self.parent_structure.radius * blob_radius_decrease_per_level

        self.num_children_blobs = num_children_blobs
        self.num_tubes = num_tubes
        
        self.child_sampler = MeshBlobSampler(
            child_radius,
            sample_children_only_inside=sample_children_only_inside
        )
        
        tube_max_radius = relative_tube_max_radius * self.parent_structure.radius
        tube_min_radius = relative_tube_min_radius * self.parent_structure.radius
        self.tube_sampler = MeshTubeSampler(tube_max_radius, tube_min_radius, self.parent_structure.radius)

    def generate(self, seed: int = None, child_blobs_batch_size: int = 1000000) -> StructurePhantom:
        rng = default_rng(seed)
        
        children_blobs = self.child_sampler.sample_children_blobs(
            self.parent_structure,
            num_children=self.num_children_blobs,
            rng=rng,
            batch_size=child_blobs_batch_size
        )

        tubes = self.tube_sampler.sample_tubes(
            parent_mesh_structure=self.parent_structure,
            num_tubes=self.num_tubes,
            rng=rng
        )
        
        return StructurePhantom(
            parent=self.parent_structure,
            children=children_blobs,
            tubes=tubes
        )
