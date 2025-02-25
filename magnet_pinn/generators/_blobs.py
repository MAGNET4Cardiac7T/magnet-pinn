import trimesh
import numpy as np
from ._perlin_noise import PerlinNoise

from typing import Union, Tuple

# blob is a deformed sphere, where the deformation is a smooth interpolation of the spheres surface offset at a set of points (knots) randomly distributed on the sphere.
class Blob:
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 seed: float = 42,
                 num_octaves: int = 2,
                 relative_disruption_strength: float = 0.3):
        
        self.center = center
        self.radius = radius
        self.relative_disruption_strength = relative_disruption_strength

        self.seed = seed
        self.num_octaves = num_octaves

        self.noise = PerlinNoise(octaves=num_octaves, seed=seed)
        self.noise_scale = np.sqrt(4/3)*self.relative_disruption_strength
        
    def generate_mesh(self, detail_level: int = 5):
        """
        Generate a mesh of the blob.
        """
        sphere = trimesh.primitives.Sphere(1, subdivisions=detail_level)

        offsets = self._calculate_offsets_at_points(sphere.vertices)
        
        vertices = (1+offsets.reshape(-1,1))*sphere.vertices
        faces = sphere.faces
        blob_mesh = trimesh.base.Trimesh(vertices, faces)
        
        # translate blob
        blob_mesh.apply_scale(self.radius)
        blob_mesh.apply_translation(self.center)
        return blob_mesh
    
    def _calculate_offsets_at_points(self, target_points: np.ndarray):
        """
        Calculate the sphere offset from the knots.
        """
        return np.array([self.noise([*point])*self.noise_scale for point in target_points])

    
class Tube:
    point: np.ndarray
    direction: np.ndarray
    radius: float = 0.1

    def __init__(self, 
                 point: np.ndarray, 
                 direction: np.ndarray, 
                 height: float = 10,
                 radius: float = 0.1):
        self.point = point
        self.direction = direction / np.linalg.norm(direction)
        self.radius = radius
        self.height = height
    
    def distance_to_line(self, line):
        """
        Calculate the distance between two lines.
        """
        # Calculate the normal vector of the plane defined by the two lines.
        normal = np.cross(self.direction, line.direction)
        if np.linalg.norm(normal) == 0:
            # The two lines are parallel.
            return np.linalg.norm(self.point - line.point)
        
        # Calculate the distance between the two lines.
        distance = np.abs(np.dot(normal, self.point - line.point)) / np.linalg.norm(normal)
        return distance
    
        # create trimesh cylinder from line

    def generate_mesh(self, 
                      detail_level: int = 5):
        """
        Create a trimesh cylinder from a line.
        """
        # Create a cylinder from the line.
        cylinder = trimesh.creation.cylinder(
            radius=self.radius,
            height=self.height,
            sections=detail_level**2,
            transform=trimesh.transformations.translation_matrix(self.point) @ trimesh.geometry.align_vectors([0, 0, 1], self.direction),
        )
        
        
        return cylinder
    

class BlobGenerator:
    def __init__(self, 
                 radius: float,
                 center: np.ndarray,
                 num_children: int,
                 num_tubes: int,
                 relative_disruption_strength: float = 0.3,
                 num_children_position_samples: int = 1000,
                 seed: int = 42):
        self.radius = radius
        self.center = center
        self.relative_disruption_strength = relative_disruption_strength
        self.num_children = num_children
        self.num_tubes = num_tubes

        self.seed = seed
        self.num_children_position_samples = num_children_position_samples

        self.rng = np.random.default_rng(seed)
        self.children_seeds = self.rng.integers(0, 2**32-1, size=num_children)
        self.tube_seeds = self.rng.integers(0, 2**32-1, size=num_tubes)

        # generate blobs
        self.children_positions = self._sample_children_positions()
        self.children_radii = self._get_children_radii(self.children_positions)
        self.children = [Blob(center=position, radius=radius, seed=int(seed)) for position, radius, seed in zip(self.children_positions, self.children_radii, self.children_seeds)]

    def _get_children_radii(self,
                            children_positions: np.ndarray):
        """
        Find the approximately maximal radii for the children.
        """

        radii_between_children = self._radii_between_points(children_positions)
        radii_to_shell = self._distance_to_shell(children_positions)

        selected_radii = np.minimum(radii_between_children, radii_to_shell)

        return selected_radii

    def _sample_children_positions(self):
        """
        Find the best candidate points.
        """

        # Sample candidate points.
        candidate_points = self._sample_points_within_ball(self.center, self.radius*(1-self.relative_disruption_strength), (self.num_children_position_samples, self.num_children))

        # calculate pairwise distances for all children positions (should yield a matrix of shape (num_candidates, num_children, num_children))
        distances = np.linalg.norm(candidate_points[:, :, None] - candidate_points[:, None], axis=-1)
        distances[distances == 0] = np.infty

        # calculate the distances to the border of the shell    
        distances_to_border = 2*(self.radius*(1-self.relative_disruption_strength) - np.linalg.norm(candidate_points, axis=-1))

        # append the distances to the border of the shell to the distances matrix
        distances = np.concatenate([distances, distances_to_border[:, :, None], ], axis=-1)

        # calculate the best candidate set
        best_candidate_idx = np.argmax(np.min(distances, axis=(-1,-2)), axis=-1)
        return candidate_points[best_candidate_idx]

        

    def _sample_points_within_ball(self, 
                                   center: np.ndarray, 
                                   radius: float,
                                   num_points: Union[int, Tuple[int]]):
        """
        Sample a point within a ball.
        """
        if isinstance(num_points, int):
            num_points = (num_points,)
    
        shape = (*num_points, len(center))

        # Sample a point on the surface of the ball.
        point = self.rng.normal(size=shape)
        point = radius*point / np.linalg.norm(point, axis=-1, keepdims=True)
        
        # Sample a point within the ball.
        point = self.rng.uniform(0, 1, size=(*num_points, 1)) ** (1 / len(center)) * point
        return point+center
    
    def _radii_between_points(self,
                              point_candidates: np.ndarray):
        distances = np.linalg.norm(point_candidates[:, None] - point_candidates[None], axis=-1)

        distances = 0.5*distances/(1+self.relative_disruption_strength)
        distances[distances==0] = np.infty
        distances = np.min(distances, axis=-1)
        return distances
    
    def _distance_to_shell(self,
                           point_candidates: np.ndarray):
        """
        Calculate the distance between the point candidates and the shell.
        """
        return (self.radius*(1-self.relative_disruption_strength) - np.linalg.norm(point_candidates - self.center, axis=-1))/(1+self.relative_disruption_strength)
        
    def generate_mesh(self,
                      detail_level: int = 5):
        """
        Generate the mesh of the blob.
        """
        scene = trimesh.scene.Scene()

        for child in self.children:
            scene.add_geometry(child.generate_mesh(detail_level=detail_level))

        return scene
    

