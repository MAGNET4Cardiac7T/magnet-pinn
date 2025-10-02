"""
    Code mostly taken from https://github.com/salaxieb/perlin_noise as it is not compatible with python >= 3.13 yet.
"""

import itertools
from functools import reduce, lru_cache
from operator import mul
from typing import Dict, Iterable, List, Optional, Tuple, Union

import math
import random

def dot(
    vec1: Union[List, Tuple],
    vec2: Union[List, Tuple],
) -> Union[float, int]:
    """Two vectors dot product.

    Parameters:
        vec1: List[float] - first vector
        vec2: List[float] - second vector

    Returns:
        Dot product of 2 vectors

    Raises:
        ValueError: if length not equal
    """
    if len(vec1) != len(vec2):
        raise ValueError("lengths of two vectors are not equal")
    return sum([val1 * val2 for val1, val2 in zip(vec1, vec2)])


def sample_vector(dimensions: int, seed: int) -> List[float]:
    """Sample normalized vector given length.

    Parameters:
        dimensions: int - space size
        seed: Optional[int] - random seed value

    Returns:
        List[float] - normalized random vector of given size
    """
    st = random.getstate()
    random.seed(seed)

    vec = []
    for _ in range(dimensions):
        vec.append(random.uniform(-1, 1))  # noqa: S311

    random.setstate(st)
    return vec


def fade(given_value: float) -> float:
    """Smoothing [0, 1] values.

    Parameters:
        given_value: float [0, 1] value for smoothing

    Returns:
        smoothed [0, 1] value

    Raises:
        ValueError: if input not in [-0.1, 1.1]
    """
    if given_value < -0.1 or given_value > 1.1:  # noqa: WPS459, WPS432
        raise ValueError("expected to have value in [-0.1, 1.1]")
    return (
        6 * math.pow(given_value, 5)  # noqa: WPS432
        - 15 * math.pow(given_value, 4)  # noqa: WPS432, W503
        + 10 * math.pow(given_value, 3)  # noqa: WPS432, W503
    )


def hasher(
    coordinates: Tuple[int, ...],
    tile_sizes: Optional[Tuple[int, ...]] = None,
) -> int:
    """Hashes coordinates to integer number and use obtained number as seed.

    Parameters:
        coordinates: Tuple[int, ...] - tuple of coordinates
        tile_sizes: Optional[Tuple[int, ...]] - optional tile sizes

    Returns:
        hash of coordinates in integer
    """
    if tile_sizes:
        coordinates = tuple(
            coors % tile for coors, tile in zip(coordinates, tile_sizes)
        )
    # fmt: off
    return max(
        1,
        int(
            abs(
                dot([10**coordinate for coordinate in range(len(coordinates))], coordinates) + 1,  # noqa: E501, WPS221
            ),
        ),
    )
    # fmt: on


def product(iterable: Union[List, Tuple]) -> float:
    """Multiplies values of iterable  each with each.

    Parameters:
        iterable: - any iterable

    Returns:
        product of values
    """
    if len(iterable) == 1:
        return iterable[0]
    return iterable[0] * product(iterable[1:])


class RandVec:
    """Vectors to give weights and contribute in final value."""

    def __init__(self, coordinates: Tuple[int, ...], seed: int):
        """Vector initializer in specified coordinates.

        Parameters:
            coordinates: Tuple[int] - vector coordinates
            seed: int - random init seed
        """
        self.coordinates = coordinates
        self.vec = sample_vector(dimensions=len(self.coordinates), seed=seed)

    def dists_to(self, coordinates: List[float]) -> Tuple[float, ...]:
        """Calculate distance to given coordinates.

        Parameters:
            coordinates: Tuplie[int] - coordinates to calculate distance

        Returns:
            distance

        """
        return tuple(
            coor1 - coor2 for coor1, coor2 in zip(coordinates, self.coordinates)
        )

    def weight_to(self, coordinates: List[float]) -> float:
        """Calculate this vector weights to given coordinates.

        Parameters:
            coordinates: Tuple[int] - target coordinates

        Returns:
            weight
        """
        # fmt: off
        weighted_dists = [fade(1 - abs(dist)) for dist in self.dists_to(coordinates)]  # noqa: E501, WPS221
        # fmt: on
        return reduce(mul, weighted_dists)

    def get_weighted_val(self, coordinates: List[float]) -> float:
        """Calculate weighted contribution of this vec to final result.

        Parameters:
            coordinates: calculate weighted relative to this coordinates

        Returns:
            weighted contribution
        """
        return self.weight_to(coordinates) * dot(
            self.vec,
            self.dists_to(coordinates),
        )
        

class PerlinNoise:
    """Smooth random noise generator.

    read more https://en.wikipedia.org/wiki/Perlin_noise
    """

    def __init__(self, octaves: float = 1, seed: Optional[int] = None):
        """Perlin Noise object initialization class.

            ex.: noise = PerlinNoise(n_dims=2, octaves=3.5, seed=777)

        Parameters:
            octaves : optional positive float, default = 1
                positive number of sub rectangles in each [0, 1] range
            seed : optional positive int, default = None
                specified seed

        Raises:
            ValueError: if seed is negative
        """
        if octaves <= 0:
            raise ValueError("octaves expected to be positive number")

        if seed is not None and not isinstance(seed, int) and seed <= 0:
            raise ValueError("seed expected to be positive integer number")

        self.octaves: float = octaves
        self.seed: int = seed if seed else random.randint(1, 10**5)  # noqa: S311, E501
        self.cache: Dict[Tuple, RandVec] = {}

    def __call__(
        self,
        coordinates: Union[int, float, List, Tuple],
        tile_sizes: Optional[Union[int, List, Tuple]] = None,
    ) -> float:
        """Forward request to noise function.

        Parameters:
            coordinates: float or list of coordinates
            tile_sizes: optional tile sizes to repetative patterns

        Returns:
            noise_value
        """
        return self.noise(coordinates, tile_sizes)

    def noise(  # noqa: WPS231
        self,
        coordinates: Union[int, float, List, Tuple, Iterable],
        tile_sizes: Optional[Union[int, List, Tuple]] = None,
    ) -> float:
        """Get perlin noise value for given coordinates.

        Parameters:
            coordinates: float or list of coordinates
            tile_sizes: optional tile sizes to repetative patterns

        Returns:
            noise_value

        Raises:
            TypeError: if coordinates is not valid type
            ValueError: if tile_sizes have different length than coordinates
        """
        if not isinstance(coordinates, (int, float, list, tuple)):
            raise TypeError("coordinates must be int, float or iterable")

        if isinstance(coordinates, (int, float)):
            coordinates = [coordinates]
        else:
            coordinates = list(coordinates)

        if tile_sizes is not None:
            if isinstance(tile_sizes, int):
                tile_sizes = [tile_sizes]
            # fmt: off
            if not all(isinstance(tile, int) for tile in tile_sizes):
                raise TypeError("tile_sizes must be int or list of int")
            # fmt: on

            if len(tile_sizes) != len(coordinates):
                raise ValueError("tile_sizes must have same length as coordinates")

            coordinates = [coors % tile for coors, tile in zip(coordinates, tile_sizes)]
            tile_sizes = tuple(tile * self.octaves for tile in tile_sizes)
        coordinates = [coordinate * self.octaves for coordinate in coordinates]

        coor_bounding_box = [
            (math.floor(coordinate), math.floor(coordinate) + 1)
            for coordinate in coordinates
        ]
        return sum(
            [
                self.get_from_cache_of_create_new(coors, tile_sizes).get_weighted_val(
                    coordinates,
                )
                for coors in itertools.product(*coor_bounding_box)
            ]  # noqa: C812
        )

    @lru_cache(maxsize=100, typed=False)  # noqa: B019
    def get_from_cache_of_create_new(
        self,
        coors: Tuple[int, ...],
        tile_sizes: Optional[Tuple[int, ...]] = None,
    ) -> RandVec:
        """Use cached RandVec or creates new.

        Parameters:
            coors: Tuple of int vector coordinates
            tile_sizes: optional tile sizes to repetative patterns

        Returns:
            RandVec
        """
        return RandVec(coors, self.seed * hasher(coors, tile_sizes))