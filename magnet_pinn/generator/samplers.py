"""
NAME
    samplers.py

DESCRIPTION
    This module provides geometric sampling strategies for phantom generation.
    Contains samplers for placing points, lines, and tubular structures within spherical
    regions while ensuring proper spatial distribution and collision avoidance. Samplers
    are configured with their operational parameters during construction and receive
    runtime parameters (RNG, iteration limits) during sampling method calls.
"""
from typing import List, Union

import numpy as np
from trimesh import Trimesh
from numpy.random import Generator
from numpy.lib.stride_tricks import sliding_window_view

try:
    from igl import fast_winding_number
except ImportError:
    from igl import fast_winding_number_for_meshes as fast_winding_number

from .utils import spheres_packable
from .structures import Tube, Blob, CustomMeshStructure
from .typing import PropertyItem, PropertyPhantom, MeshPhantom, StructurePhantom


class PropertySampler:
    """
    Sampler for generating physical properties of phantom components.
    
    Randomly samples material properties from configured distributions for each
    component of a phantom structure, enabling realistic material property
    assignment for electromagnetic simulations. Stateless sampler that receives
    RNG as parameter to sampling methods.
    """
    def __init__(self, properties_cfg):
        """
        Initialize the property sampler with configuration parameters.

        Parameters
        ----------
        properties_cfg : dict
            Configuration dictionary specifying property ranges.
            Each key should be a property name (e.g., 'conductivity') with 
            a value dict containing 'min' and 'max' keys for range bounds.
        """
        self.properties_cfg = properties_cfg

    def sample_like(self, item: Union[StructurePhantom, MeshPhantom], rng: Generator, properties_list: List = None) -> PropertyPhantom:
        """
        Sample material properties for all components of a phantom structure.

        Parameters
        ----------
        item : Union[StructurePhantom, MeshPhantom]
            The phantom structure to sample properties for. Must have parent,
            children, and tubes attributes.
        rng : Generator
            Random number generator for reproducible property sampling.
        properties_list : List, optional
            List of property names to sample. If None, samples all configured properties.

        Returns
        -------
        PropertyPhantom
            A phantom with sampled material properties for all components.
        """
        return PropertyPhantom(
            parent=self._sample(rng, properties_list),
            children=[self._sample(rng, properties_list) for _ in item.children],
            tubes=[self._sample(rng, properties_list) for _ in item.tubes]
        )

    def _sample(self, rng: Generator, properties_list: List = None):
        """
        Sample a single set of material properties.

        Parameters
        ----------
        rng : Generator
            Random number generator for reproducible property sampling.
        properties_list : List, optional
            List of property names to sample. If None, samples all configured properties.

        Returns
        -------
        PropertyItem
            A single property item with randomly sampled values.
        """
        if properties_list is None:
            properties_list = list(self.properties_cfg.keys())
        return PropertyItem(**{
            key: rng.uniform(min(dim["min"], dim["max"]), max(dim["min"], dim["max"])) 
            for key, dim in self.properties_cfg.items() 
            if key in properties_list
        })


class PointSampler:
    """
    Uniform random sampler for points within a specific spherical region.
    
    Provides methods for sampling single or multiple points uniformly distributed
    within a configured 3D ball. The sampler is bound to specific geometric 
    parameters (center and radius) at initialization.
    
    Attributes
    ----------
    center : np.ndarray
        Center coordinates of the ball as [x, y, z].
    radius : float
        Radius of the ball. Can be any numeric value including zero or negative.
    """
    
    def __init__(self, center: np.ndarray, radius: float):
        """
        Initialize point sampler for a specific spherical region.

        Parameters
        ----------
        center : np.ndarray
            Center coordinates of the ball as [x, y, z].
        radius : float
            Radius of the ball. Can be any numeric value including zero or negative.
        """
        self.center = np.array(center)
        self.radius = float(radius)
    
    def sample_point(self, rng: Generator) -> np.ndarray:
        """
        Sample a single point uniformly within the configured ball.

        Parameters
        ----------
        rng : Generator
            Random number generator for reproducible sampling.

        Returns
        -------
        np.ndarray
            Randomly sampled point within the ball as [x, y, z] coordinates.
        """
        point = rng.normal(size=self.center.shape)
        point = self.radius * point / np.linalg.norm(point)
        
        point = rng.uniform(0, 1) ** (1 / len(self.center)) * point
        return point + self.center

    def sample_points(self, num_points: int, rng: Generator) -> np.ndarray:
        """
        Sample multiple points uniformly within the configured ball.

        Parameters
        ----------
        num_points : int
            Number of points to sample. Must be positive.
        rng : Generator
            Random number generator for reproducible sampling.

        Returns
        -------
        np.ndarray
            Array of shape (num_points, 3) containing sampled coordinates.
        """
        points = rng.normal(size=(num_points, len(self.center)))
        
        points = points / np.linalg.norm(points, axis=1)[:, None]
        points = points * self.radius * rng.uniform(0, 1, size=(num_points, 1)) ** (1/len(self.center))
        points = points + self.center
        return points


class BlobSampler:
    """
    Sampler for positioning blob structures within spherical regions.
    
    Handles all blob-related sampling including hierarchical child blob placement,
    geometric constraint validation, and progressive sampling for efficient
    non-overlapping positioning. Uses sphere packing algorithms to ensure valid
    configurations and applies safety margins for surface deformation effects.
    Stateless sampler that receives RNG as parameter to sampling methods.

    Attributes
    ----------
    radius_decrease_factor : float
        Scaling factor for child blob radii relative to parent radius.
        Must be in range (0, 1) to ensure children are smaller than parent.
    """
    
    def __init__(self, radius_decrease_factor: float):
        """
        Initialize blob sampler with configuration parameters.
        
        Creates a stateless sampler that requires RNG to be passed to sampling methods.
        This design allows for proper seed control and makes the sampler reusable
        across different generation contexts.

        Parameters
        ----------
        radius_decrease_factor : float
            Scaling factor for child blob radii relative to parent radius.
            Must be in range (0, 1) to ensure children are smaller than parent.

        Raises
        ------
        ValueError
            If radius_decrease_factor is not in range (0, 1).
        """
        if radius_decrease_factor <= 0 or radius_decrease_factor >= 1:
            raise ValueError("radius_decrease_factor must be in (0, 1)")
        
        self.radius_decrease_factor = radius_decrease_factor

    def check_points_distance(self, points: np.ndarray, min_distance: float) -> bool:
        """
        Check if all points maintain minimum distance constraints.

        Parameters
        ----------
        points : np.ndarray
            Array of points with shape (n_points, 3).
        min_distance : float
            Minimum required distance between any two points.

        Returns
        -------
        bool
            True if all points are at least min_distance apart, False otherwise.
        """
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        
        distances = distances + np.eye(len(points)) * 1e+10
        
        min_distances = np.min(distances, axis=0)
        if np.any(min_distances < min_distance):
            return False
        return True

    def sample_children_blobs(self, parent_blob: Blob, num_children: int, 
                            rng: Generator, max_iterations: int = 1000000) -> list[Blob]:
        """
        Sample child blobs within a parent blob using progressive positioning.
        
        Creates appropriately sized child blobs and positions them within the parent
        using progressive sampling and geometric validation. Applies sphere packing
        constraints and safety margins to ensure realistic and collision-free placement.
        Uses empirical surface offset calculations to account for blob deformation.

        Parameters
        ----------
        parent_blob : Blob
            Parent blob structure to place children within. Must have valid
            empirical offset calculations from initialization.
        num_children : int
            Number of child blobs to generate. Must be non-negative.
            Returns empty list if zero.
        rng : Generator
            Random number generator for reproducible blob positioning and creation.
        max_iterations : int, optional
            Maximum number of sampling attempts for finding valid positions.
            Default is 1,000,000 attempts.

        Returns
        -------
        list[Blob]
            List of positioned child blob structures. Empty list if
            num_children is 0. Each blob has valid position and surface offsets.

        Raises
        ------
        RuntimeError
            If unable to place children due to geometric constraints or
            if sphere packing requirements cannot be satisfied.
        """
        if num_children == 0:
            return []
        
        child_radius = parent_blob.radius * self.radius_decrease_factor
        zero_center = np.zeros_like(parent_blob.position)
        blobs = [Blob(zero_center, child_radius, seed=rng.integers(0, 2**32-1).item()) 
                for _ in range(num_children)]

        child_radius_with_margin = self._calculate_safe_child_radius(blobs, child_radius)
        parent_allowed_radius = self._calculate_parent_sampling_radius(parent_blob, child_radius_with_margin)
        
        self._validate_packing_constraints(parent_blob, child_radius_with_margin, num_children)
        
        positions = self._find_valid_positions_progressive(
            target_positions=num_children,
            center=parent_blob.position,
            sampling_radius=parent_allowed_radius,
            min_distance=2 * child_radius_with_margin,
            rng=rng,
            max_iterations=max_iterations
        )
        
        for position, blob in zip(positions, blobs):
            blob.position = position
            
        return blobs

    def _calculate_safe_child_radius(self, blobs: list[Blob], base_radius: float) -> float:
        """
        Calculate child radius with safety margins for surface deformation.
        
        Computes the effective radius including maximum surface deformation
        to ensure collision detection accounts for blob shape irregularities.

        Parameters
        ----------
        blobs : list[Blob]
            List of child blobs with empirical offset calculations.
        base_radius : float
            Base spherical radius before deformation effects.

        Returns
        -------
        float
            Effective radius including maximum surface deformation margin.
        """
        max_offset = np.max([blob.empirical_max_offset for blob in blobs])
        return base_radius * (1 + max_offset)

    def _calculate_parent_sampling_radius(self, parent_blob: Blob, child_radius_with_margin: float) -> float:
        """
        Calculate the effective sampling radius within the parent blob.
        
        Determines the safe region for placing child blob centers, accounting
        for parent surface deformation and child size with safety margins.

        Parameters
        ----------
        parent_blob : Blob
            Parent blob with empirical surface offset calculations.
        child_radius_with_margin : float
            Child radius including surface deformation margin.

        Returns
        -------
        float
            Effective sampling radius for child blob center placement.

        Raises
        ------
        RuntimeError
            If calculated sampling radius is negative, indicating the parent
            is too small to accommodate children of the specified size.
        """
        parent_inner_radius = parent_blob.radius * (1 + parent_blob.empirical_min_offset)
        parent_allowed_radius = parent_inner_radius - child_radius_with_margin
        safety_margin = 0.02
        final_radius = parent_allowed_radius * (1 - safety_margin)
        
        if final_radius <= 0:
            raise RuntimeError(
                f"Parent blob radius {parent_blob.radius:.3f} too small to fit "
                f"child blob radius {child_radius_with_margin:.3f}"
            )
        
        return final_radius

    def _validate_packing_constraints(self, parent_blob: Blob, child_radius: float, num_children: int):
        """
        Validate that sphere packing constraints can be satisfied.
        
        Checks if the specified number of child spheres can theoretically
        be packed within the parent sphere using geometric packing limits.

        Parameters
        ----------
        parent_blob : Blob
            Parent blob defining the container volume.
        child_radius : float
            Child sphere radius including safety margins.
        num_children : int
            Number of child spheres to pack.

        Raises
        ------
        RuntimeError
            If sphere packing constraints cannot be satisfied geometrically.
        """
        parent_inner_radius = parent_blob.radius * (1 + parent_blob.empirical_min_offset)
        
        if not spheres_packable(parent_inner_radius, child_radius, num_inner=num_children):
            raise RuntimeError(
                f"Cannot pack {num_children} spheres of radius {child_radius:.3f} "
                f"into parent radius {parent_inner_radius:.3f}"
            )

    def _find_valid_positions_progressive(self, target_positions: int, center: np.ndarray, 
                                        sampling_radius: float, min_distance: float, rng: Generator,
                                        max_iterations: int = 1000000) -> np.ndarray:
        """
        Find valid non-overlapping positions using progressive batch sampling.
        
        Uses a progressive sampling strategy with increasing batch sizes to efficiently
        find the required number of non-overlapping positions within a spherical region.
        This approach balances computational efficiency with success probability by
        starting with small batches and progressively increasing batch size if needed.
        The method distributes sampling attempts across different batch sizes to
        maximize the chance of finding valid configurations.

        Parameters
        ----------
        target_positions : int
            Number of non-overlapping positions required. Must be positive.
        center : np.ndarray
            Center point of the spherical sampling region as [x, y, z] coordinates.
        sampling_radius : float
            Radius of the spherical region for position sampling. Must be positive.
        min_distance : float
            Minimum required distance between any two positions. Must be positive.
        rng : Generator
            Random number generator for reproducible position sampling.
        max_iterations : int, optional
            Maximum total number of sampling attempts across all batch sizes.
            Default is 1,000,000 attempts.

        Returns
        -------
        np.ndarray
            Array of shape (target_positions, 3) containing valid positions.
            Each row represents [x, y, z] coordinates of a position that
            satisfies the minimum distance constraint.

        Raises
        ------
        RuntimeError
            If unable to find sufficient valid positions within the maximum
            number of iterations. This typically indicates overcrowded
            configuration or insufficient sampling radius.
        """
        batch_sizes = [target_positions, target_positions * 2, target_positions * 5]
        total_attempts = 0
        
        point_sampler = PointSampler(center, sampling_radius)
        
        for batch_size in batch_sizes:
            attempts_with_batch = min(max_iterations // 3, 100000)
            
            for attempt in range(attempts_with_batch):
                candidate_positions = point_sampler.sample_points(
                    num_points=batch_size, rng=rng
                )
                
                positions_subset = candidate_positions[:target_positions]
                if self.check_points_distance(positions_subset, min_distance):
                    return positions_subset
                
                total_attempts += 1
                if total_attempts >= max_iterations:
                    break
            
            if total_attempts >= max_iterations:
                break
        
        raise RuntimeError(
            f"Could not find {target_positions} valid positions with minimum distance {min_distance:.3f} "
            f"within radius {sampling_radius:.3f} after {total_attempts} attempts. "
            f"Try reducing target_positions or increasing sampling_radius."
        )


class TubeSampler:
    """
    Multi-tube sampler with collision detection and radius variation.
    
    Generates multiple non-intersecting tubes within a spherical volume by 
    iteratively placing tubes with random radii and checking for collisions
    with previously placed structures. Stateless sampler that receives RNG
    as parameter to sampling methods.

    Attributes
    ----------
    tube_max_radius : float
        Maximum radius for generated tubes. Must be positive.
    tube_min_radius : float
        Minimum radius for generated tubes. Must be positive and less than max_radius.
    """
    
    def __init__(self, tube_max_radius: float, tube_min_radius: float, parent_radius: float = 250):
        """
        Initialize tube sampler with configuration parameters.
        
        Creates a stateless sampler that requires RNG to be passed to sampling methods.
        This design allows for proper seed control and makes the sampler reusable
        across different generation contexts. Samples tubes with random radii within
        specified bounds and ensures no intersections through collision detection. Also defines
        a fixed height for tubes based on an approximate parent radius.

        Parameters
        ----------
        tube_max_radius : float
            Maximum radius for generated tubes. Must be positive.
        tube_min_radius : float
            Minimum radius for generated tubes. Must be positive and less than max_radius.
        parent_radius : float
            Approximate parent radius to define tube height. Default is 250.

        Raises
        ------
        ValueError
            If tube radii are not positive or if min_radius >= max_radius.
        """
        if tube_max_radius <= 0:
            raise ValueError("tube_max_radius must be positive")
        if tube_min_radius <= 0:
            raise ValueError("tube_min_radius must be positive")
        if tube_min_radius >= tube_max_radius:
            raise ValueError("tube_min_radius must be less than tube_max_radius")
        
        self.tube_max_radius = tube_max_radius
        self.tube_min_radius = tube_min_radius
        self.parent_radius = parent_radius

    def _sample_line(self, center: np.ndarray, ball_radius: float, tube_radius: float, rng: Generator) -> Tube:
        """
        Sample a tube line within a ball. Defines a fixed line height as 4 times the parent radius.
        """
        point_sampler = PointSampler(center, ball_radius)
        point = point_sampler.sample_point(rng)

        direction = rng.normal(size=center.shape)
        
        center_to_point = point - center
        center_to_point_norm = np.linalg.norm(center_to_point)
        
        if center_to_point_norm > 1e-10:
            direction = direction - np.dot(direction, center_to_point) / (center_to_point_norm ** 2) * center_to_point
        
        direction = direction / np.linalg.norm(direction)
        return Tube(point, direction, tube_radius, height=4 * self.parent_radius)

    def sample_tubes(self, center: np.ndarray, radius: float, num_tubes: int, rng: Generator, max_iterations: int = 10000) -> list[Tube]:
        """
        Sample tubes within a ball with collision detection.

        Parameters
        ----------
        center : np.ndarray
            Center of the ball for tube placement.
        radius : float
            Radius of the ball for tube placement.
        num_tubes : int
            Number of tubes to generate.
        rng : Generator
            Random number generator for reproducible tube generation.
        max_iterations : int, optional
            Maximum number of attempts per tube. Default is 10,000.

        Returns
        -------
        list[Tube]
            List of non-intersecting tubes within the specified ball.
            May contain fewer tubes than requested if placement becomes impossible.
        """
        tubes = []
        for i in range(num_tubes):
            attempts = 0
            tube_placed = False
            
            while attempts < max_iterations and not tube_placed:
                tube_radius = rng.uniform(self.tube_min_radius, self.tube_max_radius)
                tube = self._sample_line(center, radius - tube_radius, tube_radius, rng)

                if np.linalg.norm(tube.position - center) + tube.radius >= radius:
                    attempts += 1
                    continue
                
                is_intersecting = False
                for existing_tube in tubes:
                    if Tube.distance_to_tube(tube, existing_tube) < tube.radius + existing_tube.radius:
                        is_intersecting = True
                        break
                
                if not is_intersecting:
                    tubes.append(tube)
                    tube_placed = True
                else:
                    attempts += 1
            
            if not tube_placed:
                break
                
        return tubes
    

class MeshBlobSampler:
    """
    Sample blobs inside a mesh volume with collision detection.

    Attributes
    ----------
    child_radius : float
        Radius of the child blobs to be placed inside the mesh. Must be positive.
    sample_children_only_inside : bool
        If True, ensures that child blobs are fully contained within the mesh volume.
    """
    
    def __init__(self, child_radius: float, sample_children_only_inside: bool = False):
        if child_radius <= 0:
            raise ValueError("child_radius must be positive")
        self.child_radius = child_radius
        self.sample_children_only_inside = sample_children_only_inside

    def _sample_inside_volume(self, mesh: Trimesh, rng: Generator, batch_size: int = 50000, points_to_return: int = 50000) -> np.ndarray:
            """
            Sample points uniformly inside the mesh volume using winding number method to check if the point is inside the mesh.

            Parameters
            ----------
            mesh : Trimesh
                The mesh to sample points inside.
            rng : Generator
                Random number generator for reproducible sampling.
            batch_size : int, optional
                Number of points to sample in batch. Default is 50,000.
            points_to_return : int, optional
                Number of points to return. Default is 50000.

            Returns
            -------
            np.ndarray
                Array of shape (points_to_return, 3) containing sampled points inside the mesh.
            """
            points = (rng.random((batch_size, 3)) * mesh.extents) + mesh.bounds[0]
            winding_number = fast_winding_number(
                mesh.vertices,
                mesh.faces,
                points
            )
            contained = np.logical_and(~ np.isclose(winding_number, 0.5), winding_number > 0.5)
            if points[contained].size > 0:
                return points[contained][:points_to_return]
            
            raise RuntimeError("Failed to sample a valid position inside the mesh")
        
    def sample_children_blobs(self, parent_mesh_structure: CustomMeshStructure, 
                            num_children: int, rng: Generator, 
                            batch_size: int = 10000000) -> list[Blob]:
        """
        Sample child blobs inside a mesh volume with collision detection.
        Places child blobs within the given mesh structure using random sampling
        and collision detection to ensure no overlaps. Optionally ensures that
        child blobs are fully contained within the mesh volume.

        Parameters
        ----------
        parent_mesh_structure : CustomMeshStructure
            The mesh structure to place child blobs within.
        num_children : int
            Number of child blobs to generate. Must be non-negative.
        rng : Generator
            Random number generator for reproducible blob positioning and creation.
        batch_size : int, optional
            Number of points to sample in batch for potential blob centers. Default is 10,000,000.

        Returns
        -------
        list[Blob]
            List of positioned child blob structures. Empty list if
            num_children is 0. Each blob has valid position.
        """
        if num_children == 0:
            return []
            
        mesh = parent_mesh_structure.mesh
        placed_blobs = [
            Blob(np.zeros(3), self.child_radius, seed=rng.integers(0, 2**32-1).item())
            for _ in range(num_children)
        ]

        potential_centers = self._sample_inside_volume(mesh, rng, batch_size=batch_size, points_to_return=batch_size)
        w = sliding_window_view(potential_centers, window_shape=num_children, axis=0)
        w = np.swapaxes(w, 1, 2)
        w_squared_norms = np.sum(w**2, axis=-1)
        dot_products = w @ w.swapaxes(-1, -2)
        squared_distances = w_squared_norms[..., :, None] + w_squared_norms[..., None, :] - 2 * dot_products
        mask = np.triu(np.ones((num_children, num_children), dtype=bool), k=1)
        D_squared = np.maximum(squared_distances, 0)
        D = np.zeros_like(squared_distances)
        D[..., mask] = np.sqrt(D_squared[..., mask])
        # n, childer, children

        effective_radii = np.array([
            blob.effective_radius for blob in placed_blobs
        ])
        centers_distances = effective_radii[:, None] + effective_radii[None, :]
        centers_distances = np.triu(centers_distances, k=1)

        valid_samples_indices = (D >= centers_distances).all(axis=(1,2))

        if np.sum(valid_samples_indices) == 0:
            raise RuntimeError("No valid blob placements found")
        
        valid_centers = w[valid_samples_indices]

        if self.sample_children_only_inside:
            # n, children, 3
            dist_to_mesh_surface = np.vectorize(parent_mesh_structure.mesh.nearest.signed_distance, signature='(n,3)->(n)')(valid_centers)
            valid_samples_indices = (dist_to_mesh_surface > effective_radii).all(axis=1)

            if valid_samples_indices.sum() == 0:
                raise RuntimeError("No valid blob placements found inside the parental mesh")
            
            valid_centers = valid_centers[valid_samples_indices]

        
        result = []
        for center, blob in zip(valid_centers[0], placed_blobs):
            blob.position = center
            result.append(blob)

        return result


class MeshTubeSampler:
    """
    Sampler for tubes inside a mesh volume with collision detection and radius variation.
    
    Allows placement of multiple tubes within a mesh while ensuring no intersections
    between tubes. Tubes are assigned random radii within specified bounds and
    are positioned using random sampling with collision checks. Stateless sampler
    that receives RNG as parameter to sampling methods.

    Attributes
    ----------
    tube_max_radius : float
        Maximum radius for generated tubes. Must be positive.
    tube_min_radius : float
        Minimum radius for generated tubes. Must be positive and less than max_radius.
    """
    def __init__(self, tube_max_radius: float, tube_min_radius: float, parent_radius: float = 250):
        if tube_max_radius <= 0:
            raise ValueError("tube_max_radius must be positive")
        if tube_min_radius <= 0:
            raise ValueError("tube_min_radius must be positive")
        if tube_min_radius >= tube_max_radius:
            raise ValueError("tube_min_radius must be less than tube_max_radius")
        self.tube_max_radius = tube_max_radius
        self.tube_min_radius = tube_min_radius
        self.parent_radius = parent_radius

    def _sample_inside_position(self, mesh: Trimesh, rng: Generator, max_iter=10000) -> np.ndarray:
        """
        Sample a single start point uniformly from the mesh volume.
        
        Parameters
        ----------
        mesh : Trimesh
            The mesh to sample points inside.
        rng : Generator
            Random number generator for reproducible sampling.
        max_iter : int, optional
            Maximum number of sampling attempts. Default is 10,000.

        Returns
        -------
        np.ndarray
            A single point inside the mesh as [x, y, z] coordinates.
        """
        for _ in range(max_iter):
            points = (rng.random((1, 3)) * mesh.extents) + mesh.bounds[0]
            contained = mesh.contains(points)
            if points[contained].size > 0:
                return points[contained][0]
        raise RuntimeError("Failed to sample a valid position inside the mesh after maximum iterations")

    def sample_tubes(self, parent_mesh_structure: CustomMeshStructure, num_tubes: int, rng: Generator,
                    max_iterations: int = 10000) -> list[Tube]:
        """
        Sample tubes inside a mesh volume with collision detection.
        Places tubes within the given mesh structure using random sampling
        and collision detection to ensure no overlaps. Defines tube height
        based on an approximate parent radius and it should be 4 times the parent radius.
        
        Parameters
        ----------
        parent_mesh_structure : CustomMeshStructure
            The mesh structure to place tubes within.
        num_tubes : int
            Number of tubes to generate. Must be non-negative.
        rng : Generator
            Random number generator for reproducible tube positioning and creation.
        max_iterations : int, optional
            Maximum number of attempts per tube. Default is 10,000.

        Returns
        -------
        list[Tube]
            List of positioned tube structures. Empty list if
            num_tubes is 0. Each tube has valid position and radius.
        """
        mesh = parent_mesh_structure.mesh
        placed_tubes: list[Tube] = []
        for i in range(num_tubes):
            attempts = 0
            tube_placed = False
            while attempts < max_iterations and not tube_placed:
                radius = rng.uniform(self.tube_min_radius, self.tube_max_radius)
                start = self._sample_inside_position(mesh, rng)
                direction = rng.normal(size=start.shape)
                # normalize direction
                direction = direction / np.linalg.norm(direction)
                tube = Tube(start, direction, radius, height=4 * self.parent_radius)
                # collision check using Tube.distance_to_tube
                is_intersecting = any(
                    Tube.distance_to_tube(tube, other) < tube.radius + other.radius
                    for other in placed_tubes
                )
                if not is_intersecting:
                    placed_tubes.append(tube)
                    tube_placed = True
                attempts += 1
            if not tube_placed:
                break
        return placed_tubes
