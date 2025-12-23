"""
NAME
    utils.py

DESCRIPTION
    This module provides utility functions for geometric calculations in phantom generation.
    Contains mathematical utilities for sphere packing validation and other geometric
    computations required during the phantom generation process.
"""
import numpy as np


def generate_fibonacci_points_on_sphere(num_points: int = 10000) -> np.ndarray:
    """
    Generate uniformly distributed points on a unit sphere using Fibonacci spiral.
    
    This function creates a nearly uniform distribution of points on the sphere surface
    using the golden angle spiral method. The resulting points are used for empirical
    sampling of surface offset variations during blob initialization. The Fibonacci
    spiral method ensures that points are distributed with nearly uniform density
    across the sphere surface, avoiding clustering at poles that occurs with some
    other spherical sampling methods. The golden angle (2π * (√5 - 1) / 2) is used
    to create the spiral pattern.
    
    Parameters
    ----------
    num_points : int, optional
        Number of points to generate on the sphere. Default is 10000.
        For blob initialization, 10,000 points provide good statistical coverage.
    
    Returns
    -------
    np.ndarray
        Array of shape (num_points, 3) containing unit sphere coordinates.
        Each row represents [x, y, z] coordinates of a point on the unit sphere.
        
    Raises
    ------
    ZeroDivisionError
        If num_points is 1, division by zero occurs in the algorithm.
    """
    if num_points <= 0:
        return np.array([]).reshape(0, 3)
    
    if num_points == 1:
        raise ZeroDivisionError("Cannot generate Fibonacci points for a single point due to division by zero")
    
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


def spheres_packable(radius_outer: float, radius_inner: float, num_inner: int = 1, safety_margin: float = 0.02) -> bool:
    """
    Check if specified number of spheres can be packed within a larger sphere.
    
    Determines whether a given number of inner spheres with specified radius
    can fit within an outer sphere without overlapping, using known geometric
    packing solutions for small numbers of spheres. The function applies analytical
    solutions for sphere packing up to 6 spheres, returning False for more than
    6 spheres as no general solution is implemented. The safety margin is applied
    to account for numerical precision and provides additional clearance between
    packed spheres.

    Parameters
    ----------
    radius_outer : float
        Radius of the outer containing sphere. Must be positive.
    radius_inner : float
        Radius of each inner sphere to be packed. Must be positive.
    num_inner : int, optional
        Number of inner spheres to pack. Default is 1. Must be positive.
        Supports packing up to 6 spheres using optimal configurations.
    safety_margin : float, optional
        Additional safety margin as fraction of inner radius. Default is 0.02.
        Applied to inner radius to ensure clearance between spheres.

    Returns
    -------
    bool
        True if the spheres can be packed without overlap, False otherwise.
        Returns False for more than 6 spheres or invalid parameters.
    """
    radius_inner = radius_inner * (1 + safety_margin)
    if num_inner == 1:
        return radius_inner <= radius_outer
    elif num_inner == 2:
        return radius_inner <= radius_outer / 2
    elif num_inner == 3:
        return radius_inner / radius_outer <= 2 * np.sqrt(3) - 3 
    elif num_inner == 4:
        return radius_inner / radius_outer <= np.sqrt(6) - 2
    elif num_inner == 5:
        return radius_inner / radius_outer <= np.sqrt(2) - 1
    elif num_inner == 6:
        return radius_inner / radius_outer <= np.sqrt(2) - 1
    else: 
        return False
