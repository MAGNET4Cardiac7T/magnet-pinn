"""
    NAME
        structure.py

    DESCRIPTION
        This module contains main meshes generative functions.
"""
from abc import ABC
from dataclasses import dataclass

import trimesh
import numpy as np
from scipy.interpolate import Rbf

from .polyhedron import icosahedron


class Structure3D(ABC):
    position: np.ndarray
    radius: float

    def __init__(self, position: np.ndarray, radius: float):
        self.position = position
        self.radius = radius

    def generate_geometry(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement `generate_geometry`")


@dataclass
class Blob(Structure3D):
    vertices: np.ndarray
    faces: np.ndarray
    num_knot_points: int
    relative_disruption_strength: float
    empirical_max_offset: float
    empirical_min_offset: float

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        num_knot_points: int = 100,
        relative_disruption_strength: float = 0.1,
    ):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.num_knot_points = num_knot_points
        self.relative_disruption_strength = relative_disruption_strength
        self._rng = np.random.default_rng()

        # Generate knot points on the unit sphere
        self.knot_points = self._generate_fibonacci_points_on_sphere()
        # Sample random offsets in [-strength, +strength]
        self.knot_offsets = self._rng.uniform(
            -self.relative_disruption_strength,
            +self.relative_disruption_strength,
            size=self.num_knot_points
        )

        self.empirical_max_offset = np.max(self.knot_offsets)
        self.empirical_min_offset = np.min(self.knot_offsets)

    def _generate_fibonacci_points_on_sphere(self, num_points: int = None) -> np.ndarray:
        if num_points is None:
            num_points = self.num_knot_points
        points = []
        phi = np.pi * (np.sqrt(5.0) - 1.0)
        for i in range(num_points):
            y = 1.0 - (2.0 * i) / (num_points - 1)
            r = np.sqrt(1.0 - y * y)
            theta = phi * i
            x = np.cos(theta) * r
            z = np.sin(theta) * r
            points.append((x, y, z))
        return np.array(points, dtype=float)

    def _interpolate_offsets(self, targets: np.ndarray) -> np.ndarray:
        """Interpolate knot_offsets onto arbitrary target points."""
        rbfi = Rbf(
            self.knot_points[:, 0],
            self.knot_points[:, 1],
            self.knot_points[:, 2],
            self.knot_offsets,
            function='gaussian',
            smooth=0.0
        )
        return rbfi(targets[:, 0], targets[:, 1], targets[:, 2])

    def generate_geometry(self, subdivisions: int = 3):
        # 1. build a unit icosphere (vertices & faces)
        #    *Here we assume a helper exists; you can replace with your own.*
        # returns (verts, faces)
        verts, faces = icosahedron(radius=1.0, detail=subdivisions)
        verts, faces = np.array(verts, dtype=float), np.array(faces, dtype=int)

        # 2. compute an offset at each vertex
        offsets = self._interpolate_offsets(verts)
        scaled_verts = (1.0 + offsets.reshape(-1, 1)) * verts

        # 3. apply global scale & translation
        scaled_verts *= self.radius
        scaled_verts += self.center

        self.vertices = scaled_verts
        self.faces = faces


@dataclass
class Tube(Structure3D):
    direction: np.ndarray
    height: float
    transform: np.ndarray
    start: np.ndarray
    end: np.ndarray
    subdivisions: int

    def __init__(self, position: np.ndarray, direction: np.ndarray, radius: float, height: float = 10000):
        super().__init__(position=position, radius=radius)
        self.direction = direction / np.linalg.norm(direction)
        self.height = height
        self.transform = (
            trimesh.transformations.translation_matrix(self.position)
            @ trimesh.geometry.align_vectors([0, 0, 1], self.direction)
        )

    @staticmethod
    def distance_to_tube(tube_1: "Tube", tube_2: "Tube") -> float:
        normal = np.cross(tube_1.direction, tube_2.direction)
        if np.linalg.norm(normal) == 0:
            return np.linalg.norm(tube_1.position - tube_2.position)
        return abs(np.dot(normal, tube_1.position - tube_2.position)) / np.linalg.norm(normal)

    def generate_geometry(self, subdivisions: int = 5) -> trimesh.Trimesh:
        half_vec = (self.direction * (self.height / 2.0))
        self.start = self.position - half_vec
        self.end   = self.position + half_vec
        self.subdivisions = subdivisions
