import trimesh
import numpy as np
from perlin_noise import PerlinNoise

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
                 relative_disruption_strength: float = 0.3,):
        self.radius = radius
        self.center = center
        self.relative_disruption_strength = relative_disruption_strength
        self.num_children = num_children
        self.num_tubes = num_tubes

        self.children_candidates = self._sample_points_within_ball(center, radius, (10000, num_children))

        # evaluate candidates 
        radii_between_children = self._radii_between_points(self.children_candidates)
        distance_to_shell = self._distance_to_shell(self.children_candidates)
        radii_per_child = np.minimum(radii_between_children, distance_to_shell)

        best_candidate_idx = np.argmin(np.min(radii_per_child, axis=-1), axis=0)
        print(best_candidate_idx)
        print(self.children_candidates[best_candidate_idx])
        print(radii_per_child[best_candidate_idx])

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
        point = np.random.normal(size=shape)
        point = radius*point / np.linalg.norm(point, axis=-1, keepdims=True)
        
        # Sample a point within the ball.
        point = np.random.uniform(0, 1, size=(*num_points, 1)) ** (1 / len(center)) * point
        return point+center
    
    def _radii_between_points(self,
                              point_candidates: np.ndarray):
        # point candidates of shape (num_candidates, num_children, 3)

        # calculate pairwise distances for all children positions (should yield a matrix of shape (num_candidates, num_children, num_children))
        distances = np.linalg.norm(point_candidates[:, :, None] - point_candidates[:, None], axis=-1)/2
        distances = distances*(1-self.relative_disruption_strength)
        distances[:, np.eye(distances.shape[1], dtype=bool)] = np.infty
        distances = np.min(distances, axis=-2)
        return distances
    
    def _distance_to_shell(self,
                           point_candidates: np.ndarray):
        """
        Calculate the distance between the point candidates and the shell.
        """
        return (self.radius*(1-self.relative_disruption_strength) - np.linalg.norm(point_candidates - self.center, axis=-1))*(1-self.relative_disruption_strength)
        
    def generate(self, **kwargs):
        blob = Blob(**kwargs)
        return blob.generate_mesh()
    
    def __call__(self, **kwargs):
        return self.generate(**kwargs)


if __name__ == "__main__":
    
    generator = BlobGenerator(radius=1, center=np.array([0,0,0]), num_children=5, num_tubes=5)
