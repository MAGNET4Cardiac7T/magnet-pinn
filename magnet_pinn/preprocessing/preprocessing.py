"""
Module for basic preprocessing functionality.
"""

import os.path as osp
from typing import List

import numpy as np
import pandas as pd
from trimesh import load_mesh
from trimesh.voxel.creation import local_voxelize

RAW_DATA_DIR_PATH = "raw"
INPUT_DIR_PATH = "input"
DIPOLES_MATERIALS_DIR_PATH = osp.join(RAW_DATA_DIR_PATH, "dipoles", "simple", "raw")
MATERIALS_FILE_NAME = "materials.txt"
STANDARD_VOXEL_SIZE = 4

E_FIELD_DATABASE_KEY = "E-FIELD"
H_FIELD_DATABASE_KEY = "H-FIELD"
POSITIONS_DATABASE_KEY = "Position"

FIELD_DIR_PATH = {E_FIELD_DATABASE_KEY: "E_field", H_FIELD_DATABASE_KEY: "H_field"}

ASCII_COLUMN_NAMES = {
    E_FIELD_DATABASE_KEY: [
        "x",
        "y",
        "z",
        "ExRe",
        "EyRe",
        "EzRe",
        "ExIm",
        "EyIm",
        "EzIm",
    ],
    H_FIELD_DATABASE_KEY: [
        "x",
        "y",
        "z",
        "HxRe",
        "HyRe",
        "HzRe",
        "HxIm",
        "HyIm",
        "HzIm",
    ],
}

ASCII_FILENAME_PATTERN = "*AC*.txt"
H5_FILENAME_PATTERN = "*AC*.h5"


class Preprocessing:
    """
    Super class for preprocessing. Contains common read-write methods.
    """

    def __init__(
        self,
        simulations_names: List[str],
        data_dir_path: str,
        voxel_size: int = STANDARD_VOXEL_SIZE,
        x_max: int = 240,
        x_min=-240,
        y_max: int = 150,
        y_min: int = -150,
        z_max: int = 150,
        z_min: int = -230,
    ) -> None:
        self.simulation_paths = list(
            map(
                lambda x: osp.join(data_dir_path, RAW_DATA_DIR_PATH, x),
                simulations_names,
            )
        )
        self.voxel_size = voxel_size
        self.positions_min = np.array([x_min, y_min, z_min])
        self.positions_max = np.array([x_max, y_max, z_max])

    def process_simulation(self, simulation_name: str, data_dir_path: str):
        pass

    def validate_coordinates(self, positions: np.array):
        data_min = np.min(positions, axis=0)
        if not np.all(self.positions_min <= data_min):
            raise Exception("Min not satisfied")

        data_max = np.max(positions, axis=0)
        if not np.all(self.positions_max >= data_max):
            raise Exception("Max not satisfied")

    def process_dipoles(self, data_dir_path: str):
        properties = self.read_properties(data_dir_path)
        meshes = (
            properties["file"].apply(lambda x: load_mesh(osp.join(data_dir_path, x)))
        ).tolist()
        mask = [
            local_voxelize(mesh, (0, 0, 0), self.voxel_size, 100) for mesh in meshes
        ]
        return properties, mask

    def read_properties(self, dir_path: str) -> pd.DataFrame:
        materials_file = osp.join(dir_path, MATERIALS_FILE_NAME)
        return pd.read_csv(materials_file)
