"""
NAME
    structures.py

DESCRIPTION
    This module contains 3D geometric structure classes for generating complex phantoms.
    It provides base classes and implementations for creating deformable organic shapes
    (blobs with Perlin noise) and cylindrical structures (tubes) used in MRI simulation.
"""
from abc import ABC
from dataclasses import dataclass

import numpy as np
from perlin_noise import PerlinNoise


class Structure3D(ABC):
    """
    Abstract base class for representing 3D geometric structures.
    
    This class defines the common interface and properties for all 3D structures
    used in phantom generation. All concrete structure implementations must inherit
    from this class and provide their specific geometric characteristics.

    Attributes
    ----------
    position : np.ndarray
        The 3D position of the structure's center as [x, y, z] coordinates.
        Must be a numpy array of shape (3,) with float values.
    radius : float
        The characteristic radius of the structure, defining its overall size.
        Must be a positive float value.
    """
    position: np.ndarray
    radius: float

    def __init__(self, position: np.ndarray, radius: float):
        self.__validate(position, radius)
        self.position = np.array(position, dtype=float)
        self.radius = float(radius)

    def __validate(self, position: np.ndarray, radius: float):
        """
        Validate input parameters for Structure3D initialization.

        Parameters
        ----------
        position : np.ndarray
            3D position vector to validate
        radius : float
            Radius value to validate

        Raises
        ------
        ValueError
            If position is not a 3D numpy array or radius is not positive
        """
        if not isinstance(position, np.ndarray) or position.shape != (3,):
            raise ValueError("Position must be a 3D numpy array.")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")


@dataclass
class Blob(Structure3D):
    """
    A deformable organic blob structure using Perlin noise for surface variation.
    
    This class creates irregular, organic-looking 3D shapes by applying Perlin noise
    to a base spherical structure. The surface deformation creates realistic tissue-like
    geometries suitable for biological phantom generation in MRI simulations. The blob
    uses Fibonacci spiral sampling to ensure uniform distribution of sample points for
    empirical offset calculation, while the Perlin noise creates smooth, natural-looking
    surface variations that are continuous and differentiable.

    Attributes
    ----------
    relative_disruption_strength : float
        Controls the amplitude of surface deformation relative to the base radius.
        Higher values create more pronounced surface irregularities.
    empirical_max_offset : float
        Maximum observed surface offset from the base sphere during initialization.
        Computed empirically from Fibonacci sphere sampling.
    empirical_min_offset : float
        Minimum observed surface offset from the base sphere during initialization.
        Used for determining safe margins in geometric operations.
    perlin_scale : float
        Standard amplitude scaling factor for Perlin noise normalization.
        Default value of 0.4 represents typical Perlin noise amplitude.
    noise : PerlinNoise
        Perlin noise generator instance used for surface deformation calculations.
    """
    relative_disruption_strength: float
    empirical_max_offset: float
    empirical_min_offset: float

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        num_octaves: int = 3,
        relative_disruption_strength: float = 0.1,
        seed: int = 42,
        perlin_scale: float = 0.4,  # Standard Perlin noise amplitude
    ):
        """
        Initialize a Blob structure with Perlin noise-based surface deformation.
        During initialization, the method generates 10,000 Fibonacci-distributed
        sample points on the unit sphere to empirically determine the maximum and
        minimum surface offsets, which are used for safe geometric operations
        and collision detection in downstream processing.

        Parameters
        ----------
        position : np.ndarray
            3D center position of the blob as [x, y, z] coordinates
        radius : float
            Base radius of the blob before surface deformation
        num_octaves : int, optional
            Number of octaves for Perlin noise generation. Higher values add
            more detailed surface features. Default is 3.
        relative_disruption_strength : float, optional
            Strength of surface deformation relative to radius. Values between
            0.05-0.2 typically produce realistic organic shapes. Default is 0.1.
        seed : int, optional
            Random seed for reproducible Perlin noise generation. Default is 42.
        perlin_scale : float, optional
            Standard amplitude scaling for Perlin noise normalization. This should
            match the typical amplitude range of the Perlin noise implementation.
            Default is 0.4.
        """
        super().__init__(position=position, radius=radius)
        self.relative_disruption_strength = relative_disruption_strength
        self.perlin_scale = perlin_scale

        self.noise = PerlinNoise(octaves=num_octaves, seed=seed)

        points = self._generate_fibonacci_points_on_sphere(num_points=10000)
        offsets_at_points = self.calculate_offsets(points)

        self.empirical_max_offset = np.max(offsets_at_points)
        self.empirical_min_offset = np.min(offsets_at_points)

    def _generate_fibonacci_points_on_sphere(self, num_points: int = None):
        """
        Generate uniformly distributed points on a unit sphere using Fibonacci spiral.
        
        This method creates a nearly uniform distribution of points on the sphere surface
        using the golden angle spiral method. The resulting points are used for empirical
        sampling of surface offset variations during blob initialization. The Fibonacci
        spiral method ensures that points are distributed with nearly uniform density
        across the sphere surface, avoiding clustering at poles that occurs with some
        other spherical sampling methods. The golden angle (2π * (√5 - 1) / 2) is used
        to create the spiral pattern.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate on the sphere. If None, uses a default value.
            For blob initialization, 10,000 points provide good statistical coverage.

        Returns
        -------
        np.ndarray
            Array of shape (num_points, 3) containing unit sphere coordinates.
            Each row represents [x, y, z] coordinates of a point on the unit sphere.
        """
        if num_points is None:
            num_points = self.num_knot_points
        
        points = []
        phi = np.pi * (np.sqrt(5.) - 1.)
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def calculate_offsets(self, vertices: np.ndarray) -> np.ndarray:
        """
        Calculate surface offset values for given vertices using Perlin noise.
        
        This method applies 3D Perlin noise to input vertices to create organic
        surface deformation. The noise values are scaled by the disruption strength
        and normalized by the Perlin scale factor to ensure consistent amplitude.
        The offset calculation follows the formula:
        offset = (perlin_noise(vertex) * relative_disruption_strength) / perlin_scale.
        The division by perlin_scale normalizes the noise to account for the typical
        amplitude range of the Perlin noise implementation, ensuring consistent
        deformation regardless of the noise generator's internal scaling.

        Parameters
        ----------
        vertices : np.ndarray
            Array of shape (N, 3) containing vertex coordinates, typically normalized
            to unit sphere coordinates. Each row represents [x, y, z] coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (N, 1) containing offset values for each vertex.
            Positive values indicate outward displacement, negative values indicate
            inward displacement from the base surface.
        """
        offsets = np.array([self.noise(list(point)) for point in vertices])
        offsets = offsets * self.relative_disruption_strength / self.perlin_scale
        offsets = offsets.reshape(-1, 1)
        return offsets


@dataclass
class Tube(Structure3D):
    """
    A cylindrical tube structure for representing vascular or tubular phantoms.
    
    This class creates infinite cylindrical structures defined by a center position,
    direction vector, and radius. Tubes are commonly used to represent blood vessels,
    airways, or other cylindrical anatomical structures in MRI phantom generation.
    Tubes are treated as infinite cylinders for geometric calculations, with the height
    parameter primarily used during mesh serialization to create finite cylinder
    meshes of appropriate length for visualization and numerical computations.

    Attributes
    ----------
    direction : np.ndarray
        Unit vector defining the tube's axis direction as [x, y, z] components.
        Automatically normalized during initialization.
    height : float
        Effective height multiplier for the tube. Used in mesh generation to control
        the cylinder length relative to the radius. Default value of 10000 creates
        effectively infinite tubes for most practical purposes.
    """
    direction: np.ndarray
    height: float

    def __init__(self, position: np.ndarray, direction: np.ndarray, radius: float, height: float = 10000):
        """
        Initialize a Tube structure with position, direction, and geometric parameters.
        The direction vector is automatically normalized to unit length, so the
        input vector magnitude does not matter - only the direction is important.

        Parameters
        ----------
        position : np.ndarray
            3D center position of the tube as [x, y, z] coordinates.
            This point lies on the tube's central axis.
        direction : np.ndarray
            3D direction vector defining the tube's axis orientation.
            Will be automatically normalized to unit length.
        radius : float
            Radius of the cylindrical tube. Must be positive.
        height : float, optional
            Height multiplier used in mesh generation. The actual cylinder height
            in mesh serialization is computed as height * radius. Default is 10000,
            creating effectively infinite tubes for most applications.
        """
        super().__init__(position=position, radius=radius)
        self.direction = direction / np.linalg.norm(direction)
        self.height = height

    @staticmethod
    def distance_to_tube(tube_1: "Tube", tube_2: "Tube") -> float:
        """
        Calculate the minimum distance between two tube centerlines.
        
        This method computes the shortest distance between two infinite cylindrical
        tubes by finding the distance between their central axes. This is used for
        collision detection and spatial relationship analysis between tubes.
        The calculation uses the formula for distance between two 3D lines:
        for parallel tubes: d = ||p1 - p2||, for non-parallel tubes: d = |n · (p1 - p2)| / ||n||,
        where n = d1 × d2 (cross product of direction vectors), p1, p2 are tube positions,
        and d1, d2 are direction vectors.

        Parameters
        ----------
        tube_1 : Tube
            First tube for distance calculation
        tube_2 : Tube
            Second tube for distance calculation

        Returns
        -------
        float
            Minimum distance between the two tube centerlines. If the tubes are
            parallel (cross product is zero), returns the distance between their
            position points.
        """
        normal = np.cross(tube_1.direction, tube_2.direction)
        if np.linalg.norm(normal) == 0:
            return np.linalg.norm(tube_1.position - tube_2.position)
        return abs(np.dot(normal, tube_1.position - tube_2.position)) / np.linalg.norm(normal)
