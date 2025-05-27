"""
NAME
    phantoms.py
DESCRIPTION
    This module contains main complex generative objects.
"""
import logging
from abc import ABC

import numpy as np
from numpy.random import Generator, default_rng

from .structures import Blob, Tube
from .typing import PhantomItem


class Phantom(ABC):
    initial_blob_radius: float
    initial_blob_center_extent: np.ndarray

    def __init__(self, initial_blob_radius: float, initial_blob_center_extent: np.ndarray):
        self.initial_blob_radius = initial_blob_radius
        self.initial_blob_center_extent = initial_blob_center_extent

    def generate(self, seed: int = None) -> PhantomItem:
        raise NotImplementedError("Subclasses should implement this method.")


class Tissue(Phantom):
    num_children_blobs: int
    blob_radius_decrease_per_level: float
    
    num_tubes: int
    tube_max_radius: float
    
    def __init__(self, 
                 num_children_blobs: int, 
                 initial_blob_radius: float, 
                 initial_blob_center_extent: np.ndarray,
                 blob_radius_decrease_per_level: float, 
                 num_tubes: int, relative_tube_max_radius: float,
                 relative_tube_min_radius: float = 0.01):
        
        super().__init__(initial_blob_radius, initial_blob_center_extent)

        self.num_children_blobs = num_children_blobs
        
        self.blob_radius_decrease_per_level = blob_radius_decrease_per_level
        self.num_tubes = num_tubes
        self.relative_tube_max_radius = relative_tube_max_radius
        self.relative_tube_min_radius = relative_tube_min_radius
        self.tube_max_radius = self.relative_tube_max_radius*initial_blob_radius
        self.tube_min_radius = self.relative_tube_min_radius*initial_blob_radius
        self.relative_disruption_strength = 0.1

    def generate(self, seed: int = None) -> PhantomItem:
        gen_object = default_rng(seed)

        initial_blob_center = np.array([
            gen_object.uniform(dim[0], dim[1])
            for _, dim in self.initial_blob_center_extent.items()
        ])

        logging.info("Generating parent blob.")
        parent_blob = Blob(initial_blob_center, self.initial_blob_radius, seed=gen_object.uniform())
        
        logging.info("Generating children blobs.")
        children_blobs = self._generate_children_blobs(parent_blob=parent_blob, gen_object=gen_object)

        logging.info("Generating tubes.")
        parent_inner_radius = parent_blob.radius*(1+parent_blob.empirical_min_offset)
        tubes = self._sample_tubes_within_ball(initial_blob_center, parent_inner_radius, self.num_tubes, self.tube_max_radius, self.tube_min_radius, gen_object=gen_object)

        return PhantomItem(
            parent=parent_blob,
            children=children_blobs,
            tubes=tubes
        )
        
    def _generate_children_blobs(self, parent_blob: Blob = None, max_iterations: int = 10000000, gen_object: Generator = None) -> list[Blob]:
        """
        Generate one hierarchy level.
        """
        if parent_blob is None:
            return [Blob(self.initial_blob_center, self.initial_blob_radius, seed=gen_object.uniform())]
        
        if self.num_children_blobs == 0:
            return []
        
        child_radius = parent_blob.radius*self.blob_radius_decrease_per_level
        zero_center = np.zeros_like(parent_blob.position)
        blobs = [Blob(zero_center, child_radius, seed=gen_object.uniform()) for _ in range(self.num_children_blobs)]
        
        min_offset_children = np.min([blob.empirical_min_offset for blob in blobs])
        max_offset_children = np.max([blob.empirical_max_offset for blob in blobs])
        
        child_radius_with_safe_margin = child_radius*(1+max_offset_children)
        
        parent_inner_radius = parent_blob.radius*(1+parent_blob.empirical_min_offset)
        # enforce additional safety margin to account for mesh sampling differences
        parent_allowed_radius = parent_inner_radius - child_radius_with_safe_margin
        safety_margin = 0.02
        parent_allowed_radius = parent_allowed_radius * (1 - safety_margin)
        
        if parent_allowed_radius < 0:
            raise Exception("Parent blob is too small to fit child blob")
        
        if not self._spheres_packable(parent_inner_radius, child_radius_with_safe_margin, num_inner=self.num_children_blobs):
            raise Exception("Sampled blobs do not fit into parent blob")
            
        idx = 0
        logging.info("Finding suitable points for child blobs. Packing ratio: {:.3f}".format(child_radius_with_safe_margin/parent_inner_radius))
        while True:
            points = self._sample_points_within_ball(parent_blob.position, parent_allowed_radius, num_points=self.num_children_blobs, gen_object=gen_object)

            # check if all points are at least child_radius_with_safe_margin away from each other
            if self._check_points_are_at_least_distance_away(points, 2*child_radius_with_safe_margin):
                break
            if idx > max_iterations:
                raise Exception("Could not find suitable points for child blob")
            idx += 1
        
        # update centers of child blobs
        for point, blob in zip(points, blobs):
            blob.position = point
            
        return blobs
    
    def _spheres_packable(self, radius_outer: float, radius_inner: float, num_inner: int = 1, safety_margin: float = 0.02):
        " Check if num_inner spheres of radius_inner can be packed into a sphere of radius_outer."
        radius_inner = radius_inner*(1+safety_margin)
        if num_inner == 1:
            return radius_inner <= radius_outer
        elif num_inner == 2:
            return radius_inner <= radius_outer/2
        elif num_inner == 3:
            return radius_inner/radius_outer <= 2 * np.sqrt(3) - 3 
        elif num_inner == 4:
            return radius_inner/radius_outer <= np.sqrt(6) - 2
        elif num_inner == 5:
            return radius_inner/radius_outer <= np.sqrt(2) - 1
        elif num_inner == 6:
            return radius_inner/radius_outer <= np.sqrt(2) - 1
        else: 
            return False
            
            
    def _check_points_are_at_least_distance_away(self, points: np.ndarray, min_distance: float):
        """
        Check if points are at least min_distance away from each other.
        """
        distances = np.linalg.norm(points[:,None,:] - points[None,:,:], axis=-1)
        
        distances = distances + np.eye(len(points))*1e+10
        
        min_distances = np.min(distances, axis=0)
        if np.any(min_distances < min_distance):
            return False
        return True

    def _sample_points_within_ball(self, center: np.ndarray, radius: float, num_points: int = 1, gen_object: Generator = None):
        """
        Sample multiple points within a ball uniformly.
        """
        points = gen_object.normal(size=(num_points, len(center)))

        points = points / np.linalg.norm(points, axis=1)[:,None]
        points = points * radius* gen_object.uniform(0, 1, size=(num_points,1))**(1/len(center))
        points = points + center
        return points

    def _sample_point_within_ball(self, center: np.ndarray, radius: float, gen_object: Generator = None):
        """
        Sample a point within a ball.
        """
        # Sample a point on the surface of the ball.
        point = gen_object.normal(size=center.shape)
        point = radius*point / np.linalg.norm(point)
        
        # Sample a point within the ball.
        point = gen_object.uniform(0, 1) ** (1 / len(center)) * point
        return point+center

    def _sample_line_within_ball(self, center: np.ndarray, ball_radius: float, tube_radius: float, gen_object: Generator = None) -> Tube:
        """
        Sample a line within a ball.
        """
        # Sample a point on the surface of the ball.
        point = self._sample_point_within_ball(center, ball_radius, gen_object=gen_object)

        # Sample a direction that is perpendicular to the radius from the center to the point.
        direction = gen_object.normal(size=center.shape)
        direction = direction - np.dot(direction, point - center) / np.linalg.norm(point - center) ** 2 * (point - center)
        direction = direction / np.linalg.norm(direction)
        return Tube(point, direction, tube_radius)


    def _sample_tubes_within_ball(self, center: np.ndarray, radius: float, num_tubes: int, tube_max_radius: float, tube_min_radius: float = 0.01, gen_object: Generator = None) -> list[Tube]:
        """
        Sample tubes within a ball.
        """
        # Sample tubes.
        tubes = []
        for i in range(num_tubes):
            while True:
                # Sample a tube.
                tube_radius = gen_object.uniform(tube_min_radius, tube_max_radius)
                tube = self._sample_line_within_ball(center, radius-tube_radius, tube_radius, gen_object=gen_object)


                # check if tube completely within the ball.
                if np.linalg.norm(tube.position - center) + tube.radius >= radius:
                    print("Tube is not completely within the ball.")
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
