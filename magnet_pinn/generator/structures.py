"""
    NAME
        structure.py

    DESCRIPTION
        This module contains main meshes generative functions.
"""
from abc import ABC
from dataclasses import dataclass

import numpy as np
from perlin_noise import PerlinNoise


class Structure3D(ABC):
    position: np.ndarray
    radius: float

    def __init__(self, position: np.ndarray, radius: float):
        self.position = np.array(position, dtype=float)
        self.radius = float(radius)


@dataclass
class Blob(Structure3D):
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
    ):
        super().__init__(position=position, radius=radius)
        self.relative_disruption_strength = relative_disruption_strength

        self.noise = PerlinNoise(octaves=num_octaves, seed=seed)

        points = self._generate_fibonacci_points_on_sphere(num_points=10000)
        offsets_at_points = self.calculate_offsets(points)

        self.empirical_max_offset = np.max(offsets_at_points)
        self.empirical_min_offset = np.min(offsets_at_points)

    def _generate_fibonacci_points_on_sphere(self, num_points: int = None):
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
        offsets = np.array([self.noise(list(point)) for point in vertices])
        offsets = offsets * self.relative_disruption_strength / 0.4
        offsets = offsets.reshape(-1, 1)
        return offsets


@dataclass
class Tube(Structure3D):
    direction: np.ndarray
    height: float

    def __init__(self, position: np.ndarray, direction: np.ndarray, radius: float, height: float = 10000):
        super().__init__(position=position, radius=radius)
        self.direction = direction / np.linalg.norm(direction)
        self.height = height

    @staticmethod
    def distance_to_tube(tube_1: "Tube", tube_2: "Tube") -> float:
        normal = np.cross(tube_1.direction, tube_2.direction)
        if np.linalg.norm(normal) == 0:
            return np.linalg.norm(tube_1.position - tube_2.position)
        return abs(np.dot(normal, tube_1.position - tube_2.position)) / np.linalg.norm(normal)
