"""
    NAME
        structure.py

    DESCRIPTION
        This module contains main meshes generative functions.
"""
from abc import ABC

import trimesh
import numpy as np
from scipy.interpolate import Rbf


class Structure(ABC):
    position: np.ndarray
    radius: float

    def __init__(self, position: np.ndarray, radius: float):
        self.position = position
        self.radius = radius

    def generate_mesh(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement `generate_mesh`")


class Blob(Structure):
    """
    Class represents a deformed sphere, where the deformation is a smooth interpolation of the spheres 
    surface offset at a set of points (knots) randomly distributed on the sphere.

    Attributes
    ----------
    position : np.ndarray
        The position of the center of the sphere.
    radius : float
        The radius of the sphere.
    num_knot_points : int
        The number of knots on the sphere.
    relative_disruption_strength : float
        The relative strength of the disruption.
    empirical_max_offset : float
        The maximum offset of the sphere surface.
    empirical_min_offset : float
        The minimum offset of the sphere surface.
    """
    num_knot_points: int
    relative_disruption_strength: float
    empirical_max_offset: float
    empirical_min_offset: float

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        num_knot_points: int = 100,
        relative_disruption_strength: float = 0.1,
    ):
        super().__init__(position=position, radius=radius)
        self.num_knot_points = num_knot_points
        self.relative_disruption_strength = relative_disruption_strength

        self.knot_vertices = self._generate_fibonacci_points_on_sphere()
        self.offsets_at_knots = np.random.uniform(
            -self.relative_disruption_strength,
            self.relative_disruption_strength,
            self.num_knot_points,
        )

        many_points = self._generate_fibonacci_points_on_sphere(num_points=10000)
        offsets = self._calculate_sphere_offset_from_knots(many_points)
        self.empirical_max_offset = np.max(offsets)
        self.empirical_min_offset = np.min(offsets)

    def generate_mesh(self, sphere_subdivisions: int = 5) -> trimesh.Trimesh:
        sphere = trimesh.primitives.Sphere(radius=1.0, subdivisions=sphere_subdivisions)
        offsets = self._calculate_sphere_offset_from_knots(sphere.vertices)
        vertices = (1 + offsets.reshape(-1, 1)) * sphere.vertices
        mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)

        mesh.apply_scale(self.radius)
        mesh.apply_translation(self.position)
        return mesh

    def _calculate_sphere_offset_from_knots(self, target_points: np.ndarray) -> np.ndarray:
        rbfi = Rbf(
            self.knot_vertices[:, 0],
            self.knot_vertices[:, 1],
            self.knot_vertices[:, 2],
            self.offsets_at_knots,
            function="gaussian",
            smooth=0.0,
        )
        return rbfi(target_points[:, 0], target_points[:, 1], target_points[:, 2])

    def _generate_fibonacci_points_on_sphere(
        self, num_points: int = None
    ) -> np.ndarray:
        if num_points is None:
            num_points = self.num_knot_points

        points = []
        phi = np.pi * (np.sqrt(5.0) - 1.0)
        for i in range(num_points):
            y = 1.0 - (i / float(num_points - 1)) * 2.0
            r = np.sqrt(1.0 - y * y)
            theta = phi * i
            x = np.cos(theta) * r
            z = np.sin(theta) * r
            points.append([x, y, z])
        return np.array(points)


class Tube(Structure):
    direction: np.ndarray

    def __init__(self, position: np.ndarray, radius: float, direction: np.ndarray):
        super().__init__(position=position, radius=radius)
        self.direction = direction / np.linalg.norm(direction)

    @staticmethod
    def distance_to_tube(tube_1: "Tube", tube_2: "Tube") -> float:
        normal = np.cross(tube_1.direction, tube_2.direction)
        if np.linalg.norm(normal) == 0:
            return np.linalg.norm(tube_1.position - tube_2.position)
        return abs(np.dot(normal, tube_1.position - tube_2.position)) / np.linalg.norm(normal)

    def generate_mesh(self, height: float = 10000, subdivisions: int = 5) -> trimesh.Trimesh:
        transform = (
            trimesh.transformations.translation_matrix(self.position)
            @ trimesh.geometry.align_vectors([0, 0, 1], self.direction)
        )
        cylinder = trimesh.creation.cylinder(
            radius=self.radius,
            height=height * self.radius,
            sections=subdivisions ** 2,
            transform=transform,
        )
        return cylinder
