"""
NAME
    voxelizing_mesh.py

DESCRIPTION
    Module for converting meshes to voxel grids
"""
from typing import Tuple

import numpy as np
from trimesh import Trimesh
from trimesh.voxel.creation import local_voxelize


class MeshVoxelizer:
    """
    The main voxelizer class.

    Attributes
    ----------
    voxel_size: float
        The size of the voxel cube.
    center: np.array
        an array with float x, y, z center coordinates
    radius: int
        a number of voxels we create in each direction
    bounds: np.array
        2 * 3 shaped array where the first row saves bottom border for cropping voxelized mesh, and the second row does the same for the top border. 

    Methods
    -------
    __init__(voxel_size, x_unique, y_unique, z_unique)
        Prepare voxelizing parameters.
    process_mesh(mesh)
        Convert the mesh to a voxel grid.
    """
    def __init__(self, 
                 voxel_size: float, 
                 x_unique: np.array, 
                 y_unique: np.array, 
                 z_unique: np.array
        ):
        """
        Prepare voxelizing parameters.

        Saves a voxel size and calculate the exact center by the extent.
        Also it defines the bottm and top borders for cropping the voxelized mesh.

        Parameters
        ----------
        voxel_size: float
            The size of the voxel cube.
        x_unique: np.array
            x grid
        y_unique: np.array
            y grid
        z_unique: np.array
            z grid
        """
        self.voxel_size = voxel_size
        self.center, self.radius, self.bounds = self.__get_center_radius_bounds__(
            x_unique, y_unique, z_unique
        )

    def __get_center_radius_bounds__(self, 
                                     x_unique: np.array,
                                     y_unique: np.array,
                                     z_unique: np.array
                                     ) -> Tuple[np.array, int, np.array]:
        """
        Claculate center and radius for voxelizing and bounds for cropping.

        The method uses unique grids x, y, z values to define the central point
        as a voxel grid center. Based on it we also define the radius as a maximum 
        distance from center to the min/max values of the grid. Then based on the center
        and radius we calculate the bounds for cropping.

        Parameters
        ----------
        x_unique: np.array
            x grid
        y_unique: np.array
            y grid
        z_unique: np.array
            z grid

        Returns
        -------
        np.array
            The center of the voxel grid
        int
            The radius of the voxel grid
        np.array
            The bounds for cropping the voxelized mesh
        """
        x_center_index = x_unique.shape[0] // 2
        y_center_index = y_unique.shape[0] // 2
        z_center_index = z_unique.shape[0] // 2

        center = np.array(
            [
                x_unique[x_center_index],
                y_unique[y_center_index],
                z_unique[z_center_index],
            ]
        ).astype(int)

        radius = max(
            (
                x_center_index,
                y_center_index,
                z_center_index,
                x_unique.shape[0] - x_center_index - 1,
                y_unique.shape[0] - y_center_index - 1,
                z_unique.shape[0] - z_center_index - 1,
            )
        )

        lows = np.array(
            [radius - x_center_index, radius - y_center_index, radius - z_center_index]
        )
        highs = lows + np.array(
            [x_unique.shape[0], y_unique.shape[0], z_unique.shape[0]]
        )
        bounds = np.row_stack([lows, highs]).astype(int)

        return center, radius, bounds

    def process_mesh(self, mesh: Trimesh) -> np.array:
        """
        Convert the mesh to a voxel grid.

        This method does the main job using predefined parameters.

        Parameters
        ----------
        mesh: trimesh.Trimesh
            The mesh to convert

        Returns
        -------
        np.array
            The voxel grid
        """
        voxel_grid = local_voxelize(
            mesh, self.center, self.voxel_size, self.radius
        ).matrix
        x_low, y_low, z_low = self.bounds[0]
        x_high, y_high, z_high = self.bounds[1]
        return voxel_grid[x_low:x_high, y_low:y_high, z_low:z_high] * 1.0
