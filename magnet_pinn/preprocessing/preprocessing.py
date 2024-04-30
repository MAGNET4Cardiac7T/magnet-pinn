"""
Module for basic preprocessing functionality.
"""

import fnmatch
import os
import os.path as osp
from typing import List

import numpy as np
import pandas as pd
from h5py import File
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

        self.dipoles_properties, self.dipoles_mask = self.process_dipoles(
            osp.join(data_dir_path, DIPOLES_MATERIALS_DIR_PATH)
        )

        self.process_simulation(simulations_names[0], data_dir_path)

    def process_simulation(self, simulation_name: str, data_dir_path: str):
        simulation_dir_path = osp.join(
            data_dir_path, RAW_DATA_DIR_PATH, simulation_name
        )
        e_field, positions = self.read_field(simulation_dir_path, E_FIELD_DATABASE_KEY)
        h_field, _ = self.read_field(simulation_dir_path, H_FIELD_DATABASE_KEY)

        object_properties, object_mask = self.process_dipoles(
            osp.join(simulation_dir_path, INPUT_DIR_PATH)
        )

    def read_field(
        self, simulation_dir_path: str, field_type: str = E_FIELD_DATABASE_KEY
    ):
        field_dir_path = osp.join(simulation_dir_path, FIELD_DIR_PATH[field_type])

        files_list = sorted(
            fnmatch.filter(os.listdir(field_dir_path), ASCII_FILENAME_PATTERN)
        )
        if len(files_list) != 0:
            positions = self.read_coordinates_from_ascii(
                osp.join(field_dir_path, files_list[0]), field_type
            )
            self.validate_coordinates(positions)
            data = list(
                map(
                    lambda i: self.read_from_ascii(
                        osp.join(field_dir_path, i), field_type
                    ),
                    files_list,
                )
            )
        else:
            positions = self.read_coordinates_from_h5(
                osp.join(field_dir_path, files_list[0])
            )
            data = list(
                map(
                    lambda i: self.read_from_h5(
                        osp.join(field_dir_path, i), field_type
                    ),
                    files_list,
                )
            )

        return data, positions

    def read_coordinates_from_ascii(
        self, file_path: str, field_type=E_FIELD_DATABASE_KEY
    ):
        columns_order = ASCII_COLUMN_NAMES[field_type]
        data = self.reorder_data_from_ascii_file(file_path, columns_order)

        return data[:, :3]

    def validate_coordinates(self, positions: np.array):
        data_min = np.min(positions, axis=0)
        if not np.all(self.positions_min <= data_min):
            raise Exception("Min not satisfied")

        data_max = np.max(positions, axis=0)
        if not np.all(self.positions_max >= data_max):
            raise Exception("Max not satisfied")

    def read_from_ascii(self, file_path: str, field_type=E_FIELD_DATABASE_KEY):
        columns_order = ASCII_COLUMN_NAMES[field_type]
        data = self.reorder_data_from_ascii_file(file_path, columns_order)

        # Extract Ex,Ey,Ez as complex
        Ex = data[:, 3] + 1j * data[:, 6]
        Ey = data[:, 4] + 1j * data[:, 7]
        Ez = data[:, 5] + 1j * data[:, 8]

        return np.column_stack((Ex, Ey, Ez))

    def reorder_data_from_ascii_file(self, file_path: str, columns_order: List[str]):
        header = np.loadtxt(file_path, max_rows=1, dtype=str)
        header = list(header[::2])

        data = np.loadtxt(file_path, skiprows=2)

        # reorder the columns
        order = [header.index(col) for col in columns_order]
        data = data[:, order]
        return data

    def read_from_h5(self, file_path: str, field_type: str = E_FIELD_DATABASE_KEY):

        with File(file_path) as f:
            values = f.get(field_type)[:]

        # Extract Ex,Ey,Ez as complex
        Ex = values["x"]["re"] + 1j * values["x"]["im"]
        Ey = values["y"]["re"] + 1j * values["y"]["im"]
        Ez = values["z"]["re"] + 1j * values["z"]["im"]

        return np.column_stack((Ex, Ey, Ez))

    def read_coordinates_from_h5(self, file_path):
        with File(file_path) as f:
            positions = f[POSITIONS_DATABASE_KEY][:]
        return np.column_stack((positions["x"], positions["y"], positions["z"])).astype(
            np.int64
        )

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
