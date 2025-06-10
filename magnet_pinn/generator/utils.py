"""
NAME
    utils.py

DESCRIPTION
    This module provides utility functions for geometric calculations in phantom generation.
    Contains mathematical utilities for sphere packing validation and other geometric
    computations required during the phantom generation process.
"""
import numpy as np


def spheres_packable(radius_outer: float, radius_inner: float, num_inner: int = 1, safety_margin: float = 0.02) -> bool:
    """
    Check if specified number of spheres can be packed within a larger sphere.
    
    Determines whether a given number of inner spheres with specified radius
    can fit within an outer sphere without overlapping, using known geometric
    packing solutions for small numbers of spheres.

    Parameters
    ----------
    radius_outer : float
        Radius of the outer containing sphere. Must be positive.
    radius_inner : float
        Radius of each inner sphere to be packed. Must be positive.
    num_inner : int, optional
        Number of inner spheres to pack. Default is 1. Must be positive.
    safety_margin : float, optional
        Additional safety margin as fraction of inner radius. Default is 0.02.

    Returns
    -------
    bool
        True if the spheres can be packed, False otherwise.

    Notes
    -----
    Uses analytical solutions for sphere packing up to 6 spheres.
    For more than 6 spheres, returns False as no general solution is implemented.
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
