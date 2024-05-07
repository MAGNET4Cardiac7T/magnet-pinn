"""
Module for basic preprocessing functionality.
"""

import os.path as osp
from typing import List

import numpy as np

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY,
    FieldReader,
)

RAW_DATA_DIR_PATH = "raw"
DIPOLES_MATERIALS_DIR_PATH = osp.join(RAW_DATA_DIR_PATH, "dipoles", "simple", "raw")
STANDARD_VOXEL_SIZE = 4


class Preprocessing:
    """
    Super class for preprocessing. Contains common read-write methods.
    """

    def __init__(
        self,
        simulations_names: List[str],
        data_dir_path: str,
        voxel_size: int = STANDARD_VOXEL_SIZE,
        **kwargs,
    ) -> None:
        self.simulations_dir_path = osp.join(data_dir_path, RAW_DATA_DIR_PATH)
        self.voxel_size = voxel_size
        self.positions_min = np.array(
            (kwargs["x_min"], kwargs["y_min"], kwargs["z_min"])
        )
        self.positions_max = np.array(
            (kwargs["x_max"], kwargs["y_max"], kwargs["z_max"])
        )

        list(map(lambda x: self.process_simulation(x), simulations_names))

    def process_simulation(self, simulation_name: str):
        simulation_dir_path = osp.join(self.simulations_dir_path, simulation_name)

        e_field_values, positions = self.read_field(
            simulation_dir_path, E_FIELD_DATABASE_KEY
        )
        h_field_values, _ = self.read_field(simulation_dir_path, H_FIELD_DATABASE_KEY)

        print(f"Simulation {simulation_name} has been processed")

    def read_field(
        self, simulation_dir_path: str, field_type: str = E_FIELD_DATABASE_KEY
    ):
        field_values = []
        positions = None
        reader = FieldReader(simulation_dir_path, field_type)
        for data, pos in reader.read_data():
            if self.sanity_check(pos):
                field_values.append(data)
                if positions is None:
                    positions = pos
        return field_values, positions

    def sanity_check(self, positions: np.array):
        data_min = np.min(positions, axis=0)
        if not np.all(self.positions_min <= data_min):
            raise Exception("Min not satisfied")

        data_max = np.max(positions, axis=0)
        if not np.all(self.positions_max >= data_max):
            raise Exception("Max not satisfied")

        return True
