from typing import Tuple

import numpy as np
from trimesh.voxel.creation import local_voxelize


class MeshVoxelizer:
    def __init__(self, positions: np.array, voxel_size: int):
        self.voxel_size = voxel_size
        self.center, self.radius, self.bounds = self.__get_center_radius_bounds__(
            positions
        )

    def __get_center_radius_bounds__(self, positions: np.array):
        x_unique = np.unique(positions[:, 0])
        y_unique = np.unique(positions[:, 1])
        z_unique = np.unique(positions[:, 2])

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

    def process_mesh(self, mesh):
        voxel_grid = local_voxelize(
            mesh, self.center, self.voxel_size, self.radius
        ).matrix
        x_low, y_low, z_low = self.bounds[0]
        x_high, y_high, z_high = self.bounds[1]
        return voxel_grid[x_low:x_high, y_low:y_high, z_low:z_high] * 1.0
