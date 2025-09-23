"""
NAME
    voxelizing_mesh.py

DESCRIPTION
    Module for converting meshes to voxel grids
"""
from typing import Tuple

import numpy as np
import numpy.typing as npt
from trimesh import Trimesh
from trimesh.voxel.creation import local_voxelize
try:
    from igl import fast_winding_number
except ImportError:
    from igl import fast_winding_number_for_meshes as fast_winding_number
import einops


class MeshVoxelizer:
    """
    Voxelizer class.

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

    Attributes
    ----------
    voxel_size: float
        The size of the voxel cube.
    x: np.array
        x grid
    y: np.array
        y grid
    z: np.array
        z grid
    points: np.array
        The points in the grid 

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

        self.__validate_input(x_unique, "x")
        self.__validate_input(y_unique, "y")
        self.__validate_input(z_unique, "z")

        self.x = x_unique
        self.y = y_unique
        self.z = z_unique

        x_grid, y_grid, z_grid = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')
        self.points = np.stack([x_grid, y_grid, z_grid], axis=-1)
        self.points = np.ascontiguousarray(einops.rearrange(self.points, 'x y z d -> (x y z) d'))

    def __validate_input(self, grid: npt.NDArray[np.float64], axis: str):
        """
        Validate the input grid for voxelization.

        Parameters
        ----------
        grid: npt.NDArray[np.float64]
            The grid to validate.
        axis: str
            The axis of the grid (x, y, or z).
        
        Raises
        ------
        ValueError
            If the grid is not sorted in ascending order or does not match the expected voxel size.
        """
        if grid[0] >= grid[-1]:
            raise ValueError("Grid must be sorted in ascending order.")
        
        steps = ((grid[-1] - grid[0]) / self.voxel_size).astype(int) + 1
        supposed_grid = np.linspace(grid[0], grid[-1], steps)

        if not np.equal(grid, supposed_grid).all():
            raise ValueError(f"Invalid {axis} grid {grid} for the {self.voxel_size} vixel size.")

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
        vertices = np.ascontiguousarray(mesh.vertices)
        faces = np.ascontiguousarray(mesh.faces)
        points = np.ascontiguousarray(self.points, dtype=vertices.dtype)

        winding_number = fast_winding_number(
            vertices, 
            faces, 
            points
        )

        mask = np.logical_and(
            ~ np.isclose(winding_number, 0.5),
            winding_number > 0.5
        )
        mask = np.ascontiguousarray(
            einops.rearrange(
                mask, '(x y z) -> x y z',
                x=len(self.x), y=len(self.y), z=len(self.z)
            )
        ).astype(np.uint8)
        return mask
