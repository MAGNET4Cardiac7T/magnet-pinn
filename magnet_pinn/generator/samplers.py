"""
NAME
    samplers.py

DESCRIPTION
    This module provides geometric sampling strategies for phantom generation.
    Contains samplers for placing points, lines, and tubular structures within spherical
    regions while ensuring proper spatial distribution and collision avoidance.
"""
import numpy as np
from numpy.random import Generator

from .structures import Tube, Blob
from .utils import spheres_packable


class PointSampler:
    """
    Uniform random sampler for points within spherical regions.
    
    Provides methods for sampling single or multiple points uniformly distributed
    within 3D balls, with utilities for checking minimum distance constraints
    between sampled points. Stateless sampler that receives RNG as parameter.
    """
    
    def sample_point(self, center: np.ndarray, radius: float, rng: Generator) -> np.ndarray:
        """Sample a single point within a ball uniformly."""
        # Sample a point on the surface of the ball.
        point = rng.normal(size=center.shape)
        point = radius * point / np.linalg.norm(point)
        
        # Sample a point within the ball.
        point = rng.uniform(0, 1) ** (1 / len(center)) * point
        return point + center

    def sample_points(self, center: np.ndarray, radius: float, num_points: int, rng: Generator) -> np.ndarray:
        """Sample multiple points within a ball uniformly."""
        points = rng.normal(size=(num_points, len(center)))
        
        points = points / np.linalg.norm(points, axis=1)[:, None]
        points = points * radius * rng.uniform(0, 1, size=(num_points, 1)) ** (1/len(center))
        points = points + center
        return points

    def check_points_distance(self, points: np.ndarray, min_distance: float) -> bool:
        """Check if points are at least min_distance away from each other."""
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        
        distances = distances + np.eye(len(points)) * 1e+10
        
        min_distances = np.min(distances, axis=0)
        if np.any(min_distances < min_distance):
            return False
        return True


class BlobSampler:
    """
    Sampler for positioning blob structures within spherical regions.
    
    Handles all blob-related sampling including hierarchical child blob placement,
    geometric constraint validation, and progressive sampling for efficient
    non-overlapping positioning. Uses sphere packing algorithms to ensure valid
    configurations and applies safety margins for surface deformation effects.
    Stateless sampler that receives RNG as parameter to sampling methods.
    """
    
    def __init__(self):
        """
        Initialize blob sampler.
        
        Creates a stateless sampler that requires RNG to be passed to sampling methods.
        This design allows for proper seed control and makes the sampler reusable
        across different generation contexts.
        """
        self.point_sampler = PointSampler()

    def sample_children_blobs(self, parent_blob: Blob, num_children: int, 
                            radius_decrease_factor: float, rng: Generator, 
                            max_iterations: int = 1000000) -> list[Blob]:
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
        radius_decrease_factor : float
            Scaling factor for child blob radii relative to parent radius.
            Must be in range (0, 1) to ensure children are smaller than parent.
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
        
        # Calculate child blob parameters
        child_radius = parent_blob.radius * radius_decrease_factor
        zero_center = np.zeros_like(parent_blob.position)
        blobs = [Blob(zero_center, child_radius, seed=rng.integers(0, 2**32-1).item()) 
                for _ in range(num_children)]

        # Calculate geometric constraints with safety margins
        child_radius_with_margin = self._calculate_safe_child_radius(blobs, child_radius)
        parent_allowed_radius = self._calculate_parent_sampling_radius(parent_blob, child_radius_with_margin)
        
        # Validate sphere packing constraints
        self._validate_packing_constraints(parent_blob, child_radius_with_margin, num_children)
        
        # Find valid positions using progressive sampling
        positions = self._find_valid_positions_progressive(
            target_positions=num_children,
            center=parent_blob.position,
            sampling_radius=parent_allowed_radius,
            min_distance=2 * child_radius_with_margin,
            rng=rng,
            max_iterations=max_iterations
        )
        
        # Assign positions to child blobs
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
        # Progressive batch sizes for improved efficiency
        batch_sizes = [target_positions, target_positions * 2, target_positions * 5]
        total_attempts = 0
        
        for batch_size in batch_sizes:
            # Distribute iterations across batch sizes
            attempts_with_batch = min(max_iterations // 3, 100000)
            
            for attempt in range(attempts_with_batch):
                # Sample batch of candidate positions
                candidate_positions = self.point_sampler.sample_points(
                    center, sampling_radius, num_points=batch_size, rng=rng
                )
                
                # Check if first target_positions satisfy distance constraints
                positions_subset = candidate_positions[:target_positions]
                if self.point_sampler.check_points_distance(positions_subset, min_distance):
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
    """
    
    def __init__(self):
        """
        Initialize tube sampler.
        
        Creates a stateless sampler that requires RNG to be passed to sampling methods.
        This design allows for proper seed control and makes the sampler reusable
        across different generation contexts.
        """
        self.point_sampler = PointSampler()

    def _sample_line(self, center: np.ndarray, ball_radius: float, tube_radius: float, rng: Generator) -> Tube:
        """Sample a tube line within a ball (formerly LineSampler functionality)."""
        # Sample a point on the surface of the ball.
        point = self.point_sampler.sample_point(center, ball_radius, rng)

        # Sample a direction that is perpendicular to the radius from the center to the point.
        direction = rng.normal(size=center.shape)
        direction = direction - np.dot(direction, point - center) / np.linalg.norm(point - center) ** 2 * (point - center)
        direction = direction / np.linalg.norm(direction)
        return Tube(point, direction, tube_radius)

    def sample_tubes(self, center: np.ndarray, radius: float, num_tubes: int, 
                     tube_max_radius: float, tube_min_radius: float, rng: Generator) -> list[Tube]:
        """Sample tubes within a ball with collision detection."""
        tubes = []
        for i in range(num_tubes):
            while True:
                # Sample a tube.
                tube_radius = rng.uniform(tube_min_radius, tube_max_radius)
                tube = self._sample_line(center, radius - tube_radius, tube_radius, rng)

                # check if tube completely within the ball.
                if np.linalg.norm(tube.position - center) + tube.radius >= radius:
                    continue
                
                # Check if the tube intersects with any existing tube.
                is_intersecting = False
                for existing_tube in tubes:
                    if Tube.distance_to_tube(tube, existing_tube) < tube.radius + existing_tube.radius:
                        is_intersecting = True
                        break
                if not is_intersecting:
                    break
            tubes.append(tube)
        return tubes
