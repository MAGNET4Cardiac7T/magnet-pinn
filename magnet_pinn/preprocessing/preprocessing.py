"""
Module for basic preprocessing functionality.
"""

import os.path as osp
from os import makedirs
from typing import Tuple

import numpy as np
import pandas as pd
from h5py import File
from tqdm import tqdm

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY,
    FieldReader,
)
from magnet_pinn.preprocessing.reading_properties import PropertyReader
from magnet_pinn.preprocessing.voxelizing_mesh import MeshVoxelizer

RAW_DATA_DIR_PATH = "raw"
DIPOLES_MATERIALS_DIR_PATH = osp.join("dipoles", "simple", "raw")
INPUT_DIR_PATH = "Input"
PROCESSED_DIR_PATH = "processed"
STANDARD_VOXEL_SIZE = 4
FEATURE_NAMES = ("conductivity", "permittivity", "density")
AIR_FEATURES = {"conductivity": 0.0, "permittivity": 1.0006, "density": 1.293}
AIR_FEATURE_VALUES = tuple(AIR_FEATURES[feature_name] for feature_name in FEATURE_NAMES)


class Preprocessing:
    """
    Super class for preprocessing. Contains common read-write methods.
    """

    def __init__(
        self,
        data_dir_path: str,
        **kwargs,
    ) -> None:
        self.data_dir_path = data_dir_path
        self.simulations_dir_path = osp.join(data_dir_path, RAW_DATA_DIR_PATH)

        self.dipoles_properties, self.dipoles_meshes = self.__get_dipoles_data()

        self.positions_min = np.array(
            (kwargs["x_min"], kwargs["y_min"], kwargs["z_min"])
        )
        self.positions_max = np.array(
            (kwargs["x_max"], kwargs["y_max"], kwargs["z_max"])
        )

    def __get_dipoles_data(self):
        dipoles_properties_dir_path = osp.join(
            self.simulations_dir_path, DIPOLES_MATERIALS_DIR_PATH
        )
        dipoles_property_reader = PropertyReader(dipoles_properties_dir_path)
        dipoles_meshes = dipoles_property_reader.read_meshes()
        return dipoles_property_reader.properties, dipoles_meshes

    def process_simulations(self, simulation_names: str):
        for simulation_name in tqdm(simulation_names):
            self.__process_simulation(simulation_name)

    def __process_simulation(self, simulation_name: str):
        simulation_dir_path = osp.join(self.simulations_dir_path, simulation_name)

        e_field_reader = FieldReader(simulation_dir_path, E_FIELD_DATABASE_KEY)
        h_field_reader = FieldReader(simulation_dir_path, H_FIELD_DATABASE_KEY)

        if not np.array_equal(e_field_reader.positions, h_field_reader.positions):
            raise Exception("Different positions for the E-field and H-field")
        
        self.__sanity_check(e_field_reader.positions)

        object_properties, object_meshes = self.__get_objects_data(simulation_dir_path)
        dipoles_masks, object_masks = self._get_masks(object_meshes, e_field_reader.positions)

        features = self.__calculate_features(
            dipoles_masks, object_masks, object_properties
        )

        e_field_values = e_field_reader.read_data()
        h_field_values = h_field_reader.read_data()

        self._format_and_write_dataset(
            simulation_name,
            features,
            e_field_values,
            h_field_values,
            e_field_reader.positions,
            object_masks,
        )

    def __sanity_check(self, positions: np.array):
        data_min = np.min(positions, axis=0)
        if not np.all(self.positions_min <= data_min):
            raise Exception("Min not satisfied")

        data_max = np.max(positions, axis=0)
        if not np.all(self.positions_max >= data_max):
            raise Exception("Max not satisfied")

    def __get_objects_data(self, simulation_dir_path: str):
        properties_dir_path = osp.join(simulation_dir_path, INPUT_DIR_PATH)
        property_reader = PropertyReader(properties_dir_path)
        object_meshes = property_reader.read_meshes()
        return property_reader.properties, object_meshes

    def _get_masks(self, objects_meshes, positions: np.array):
        raise NotImplementedError()

    def __calculate_features(
        self,
        dipoles_masks: np.array,
        objects_masks: np.array,
        object_properties: pd.DataFrame,
    ):
        object_properties_values = object_properties.loc[:, FEATURE_NAMES].to_numpy().T
        dipoles_properties_values = (
            self.dipoles_properties.loc[:, FEATURE_NAMES].to_numpy().T
        )

        """
        Masks arrays have shape (x, y, z, n), where n is number of dipoles/objects.
        We need to create a new axis for the features. It will be pre-last.
        """

        object_properties_values = object_properties.loc[:, FEATURE_NAMES].to_numpy().T
        extended_object_masks = np.repeat(
            np.expand_dims(objects_masks, axis=-2), len(FEATURE_NAMES), axis=-2
        )
        objects_features = np.sum(
            extended_object_masks * object_properties_values, axis=-1
        )

        dipoles_properties_values = (
            self.dipoles_properties.loc[:, FEATURE_NAMES].to_numpy().T
        )
        dipoles_extended_masks = np.repeat(
            np.expand_dims(dipoles_masks, axis=-2), len(FEATURE_NAMES), axis=-2
        )
        dipoles_features = np.sum(
            dipoles_extended_masks * dipoles_properties_values, axis=-1
        )

        features = dipoles_features + objects_features

        # set air feaure values
        features[np.sum(features, axis=-1) == 0, :] = AIR_FEATURE_VALUES

        general_mask = np.sum(
            np.concatenate((dipoles_masks, objects_masks), axis=-1), axis=-1
        ).astype(bool)
        features = np.concatenate(
            (features, np.expand_dims(general_mask, axis=-1)), axis=-1
        )
        return features

    def _format_and_write_dataset(
        self,
        simulation_name: str,
        features: np.array,
        e_field: np.array,
        h_field: np.array,
        positions: np.array,
        object_masks: np.array,
    ):
        raise NotImplementedError()


class GridPreprocessing(Preprocessing):
    def __init__(
        self, data_dir_path: str, voxel_size: int = STANDARD_VOXEL_SIZE, **kwargs
    ):
        super().__init__(data_dir_path, **kwargs)
        self.voxel_size = voxel_size

    def _get_masks(self, objects_meshes, positions: np.array):
        voxelizer = MeshVoxelizer(positions, self.voxel_size)
        dipoles_masks = list(
            map(
                lambda x: voxelizer.process_mesh(x),
                self.dipoles_meshes,
            )
        )
        object_masks = list(
            map(
                lambda x: voxelizer.process_mesh(x),
                objects_meshes,
            )
        )

        return (np.stack(dipoles_masks, axis=-1), np.stack(object_masks, axis=-1))

    def _format_and_write_dataset(
        self,
        simulation_name: str,
        features: np.array,
        e_field: np.array,
        h_field: np.array,
        positions: np.array,
        object_masks: np.array,
    ):
        target_dir_name = f"grid_processed_voxel_size_{self.voxel_size}"
        target_dir_path = osp.join(
            self.data_dir_path, PROCESSED_DIR_PATH, target_dir_name
        )
        makedirs(target_dir_path, exist_ok=True)

        target_file_name = f"{simulation_name}.h5"
        output_file_path = osp.join(target_dir_path, target_file_name)

        e_field = e_field.reshape((
            object_masks.shape[0],
            object_masks.shape[1],
            object_masks.shape[2],
            3,
            e_field.shape[-1],
        ))

        h_field = h_field.reshape((
            object_masks.shape[0],
            object_masks.shape[1],
            object_masks.shape[2],
            3,
            h_field.shape[-1],
        ))

        with File(output_file_path, "w") as f:
            f.create_dataset("input", data=features)
            f.create_dataset("efield", data=e_field)
            f.create_dataset("hfield", data=h_field)
            f.create_dataset("subject", data=object_masks)


class GraphPreprocessing(Preprocessing):

    def _get_masks(self, objects_meshes, positions: np.array):

        object_masks = list(map(lambda x: x.contains(positions), objects_meshes))

        dipoles_masks = list(
            map(lambda x: x.contains(positions), self.dipoles_meshes)
        )

        return (np.stack(dipoles_masks, axis=-1), np.stack(object_masks, axis=-1))

    def _format_and_write_dataset(
        self,
        simulation_name: str,
        features: np.array,
        e_field: np.array,
        h_field: np.array,
        positions: np.array,
        object_masks: np.array,
    ):
        target_dir_name = "graph_processed"
        target_dir_path = osp.join(
            self.data_dir_path, PROCESSED_DIR_PATH, target_dir_name
        )
        makedirs(target_dir_path, exist_ok=True)

        target_file_name = f"{simulation_name}.h5"
        output_file_path = osp.join(target_dir_path, target_file_name)

        e_field = e_field.reshape(-1, 3)
        h_field = h_field.reshape(-1, 3)

        with File(output_file_path, "w") as f:
            f.create_dataset("input", data=features)
            f.create_dataset("efield", data=e_field)
            f.create_dataset("hfield", data=h_field)
            f.create_dataset("subject", data=object_masks)
            f.create_dataset("positions", positions)
