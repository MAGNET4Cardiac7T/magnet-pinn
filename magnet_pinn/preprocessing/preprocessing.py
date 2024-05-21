"""
Module for basic preprocessing functionality.
"""

import os.path as osp
from typing import List

import numpy as np
import pandas as pd
from h5py import File
from trimesh.voxel.creation import local_voxelize

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY,
    FieldReader,
)
from magnet_pinn.preprocessing.reading_properties import PropertyReader

RAW_DATA_DIR_PATH = "raw"
DIPOLES_MATERIALS_DIR_PATH = osp.join("dipoles", "simple", "raw")
INPUT_DIR_PATH = "input"
PROCESSED_DIR_PATH = "processed"
STANDARD_VOXEL_SIZE = 4
FEATURE_NAMES = ("conductivity", "permittivity", "density")
AIR_FEATURES = {"conductivity": 0.0, "permittivity": 1.0006, "density": 1.293}


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
        self.data_dir_path = data_dir_path
        self.simulations_dir_path = osp.join(data_dir_path, RAW_DATA_DIR_PATH)

        self.dipoles_properties, self.dipoles_meshes = self.__get_dipoles_data__()

        self.voxel_size = voxel_size
        self.positions_min = np.array(
            (kwargs["x_min"], kwargs["y_min"], kwargs["z_min"])
        )
        self.positions_max = np.array(
            (kwargs["x_max"], kwargs["y_max"], kwargs["z_max"])
        )

        for i in simulations_names:
            self.process_simulation(i)
            break

    def __get_dipoles_data__(self):
        dipoles_properties_dir_path = osp.join(
            self.simulations_dir_path, DIPOLES_MATERIALS_DIR_PATH
        )
        dipoles_property_reader = PropertyReader(dipoles_properties_dir_path)
        dipoles_meshes = dipoles_property_reader.read_meshes()
        return dipoles_property_reader.properties, dipoles_meshes

    def process_simulation(self, simulation_name: str):
        simulation_dir_path = osp.join(self.simulations_dir_path, simulation_name)

        e_field_reader = FieldReader(simulation_dir_path, E_FIELD_DATABASE_KEY)
        h_field_reader = FieldReader(simulation_dir_path, H_FIELD_DATABASE_KEY)

        if not np.array_equal(e_field_reader.coordinates, h_field_reader.coordinates):
            raise Exception("Different coordinates for the E-field and H-field")
        self.__sanity_check__(e_field_reader.coordinates)
        positions = e_field_reader.coordinates

        center, radius, bounds = self.__get_center_radius_bounds__(positions)

        object_properties, object_masks = self.__get_objects_data__(
            simulation_dir_path, center, radius, bounds
        )

        dipoles_masks = list(
            map(
                lambda x: self.__process_mesh__(x, center, radius, bounds),
                self.dipoles_meshes,
            )
        )

        input_features = self.__calculate_features__(
            dipoles_masks, object_masks, object_properties
        )
        general_mask = sum(dipoles_masks + object_masks)
        final_features = np.stack((general_mask, *input_features))

        e_field_values = e_field_reader.read_data()
        h_field_values = h_field_reader.read_data()

        self.__format_and_write__(
            simulation_name,
            final_features,
            e_field_values,
            h_field_values,
            positions,
            object_masks,
        )

        print(f"Simulation {simulation_name} processed")

    def __sanity_check__(self, positions: np.array):
        data_min = np.min(positions, axis=0)
        if not np.all(self.positions_min <= data_min):
            raise Exception("Min not satisfied")

        data_max = np.max(positions, axis=0)
        if not np.all(self.positions_max >= data_max):
            raise Exception("Max not satisfied")

        return True

    def __get_center_radius_bounds__(self, coordinates: np.array):
        x_unique = np.unique(coordinates[:, 0])
        y_unique = np.unique(coordinates[:, 1])
        z_unique = np.unique(coordinates[:, 2])

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

    def __get_objects_data__(
        self, simulation_dir_path: str, center: np.array, radius: int, bounds: np.array
    ):
        properties_dir_path = osp.join(simulation_dir_path, INPUT_DIR_PATH)
        property_reader = PropertyReader(properties_dir_path)
        meshes = property_reader.read_meshes()
        masks = list(
            map(lambda x: self.__process_mesh__(x, center, radius, bounds), meshes)
        )
        return property_reader.properties, masks

    def __process_mesh__(self, mesh, center, radius, bounds):
        voxel_grid = local_voxelize(mesh, center, self.voxel_size, radius).matrix
        x_low, y_low, z_low = bounds[0]
        x_high, y_high, z_high = bounds[1]
        return voxel_grid[x_low:x_high, y_low:y_high, z_low:z_high] * 1.0

    def __calculate_features__(
        self,
        dipoles_masks: List[np.array],
        object_masks: List[np.array],
        object_properties: pd.DataFrame,
    ):
        features = []
        for feature_name in FEATURE_NAMES:

            dipoles_features = sum(
                (
                    dipoles_masks[i] * feature
                    for i, feature in enumerate(self.dipoles_properties[feature_name])
                )
            )

            object_features = sum(
                (
                    object_masks[i] * feature
                    for i, feature in enumerate(object_properties[feature_name])
                )
            )

            feature_values = dipoles_features + object_features

            air_mask = 1 - sum(dipoles_masks + object_masks)
            feature_values += air_mask * AIR_FEATURES[feature_name]
            features.append(feature_values)

        return features

    def __format_and_write__(
        self,
        simulation_name: str,
        features: np.array,
        e_field: np.array,
        h_field: np.array,
        positions: np.array,
        object_masks: List[np.array],
    ):
        reshaped_efield = self.__format_field_values__(e_field, positions)
        reshaped_hfield = self.__format_field_values__(h_field, positions)

        target_file_name = f"{simulation_name}-voxel_size_{self.voxel_size}.h5"
        output_file_path = osp.join(
            self.data_dir_path, PROCESSED_DIR_PATH, target_file_name
        )
        with File(output_file_path, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("e_field", data=reshaped_efield)
            f.create_dataset("h_field", data=reshaped_hfield)
            f.create_dataset("object_masks", data=np.stack(object_masks))

    def __format_field_values__(self, field_values: np.array, positions: np.array):
        x_axis_size = np.unique(positions[:, 0]).shape[0]
        y_axis_size = np.unique(positions[:, 1]).shape[0]
        z_axis_size = np.unique(positions[:, 2]).shape[0]
        new_shape = (x_axis_size, y_axis_size, z_axis_size)
        fields = []
        for field in field_values:
            field_reshaped = np.empty(
                (x_axis_size, y_axis_size, z_axis_size, 3), dtype=np.complex128
            )
            field_reshaped[:, :, :, 0] = np.reshape(field[:, 0], new_shape, order="F")
            field_reshaped[:, :, :, 1] = np.reshape(field[:, 1], new_shape, order="F")
            field_reshaped[:, :, :, 2] = np.reshape(field[:, 2], new_shape, order="F")
            fields.append(field_reshaped)
        return np.stack(fields, axis=-1)
