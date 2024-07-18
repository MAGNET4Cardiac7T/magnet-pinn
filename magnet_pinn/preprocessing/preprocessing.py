"""
NAME
    preprocessing.py

DESCRIPTION
    This module contains preprocessing classes for the simulation data.

CLASSES
    Preprocessing
    GridPreprocessing
    GraphPreprocessing
"""

import os.path as osp
from typing import Tuple, List
from os import makedirs, listdir
from abc import ABC, abstractmethod

import numpy as np
from numpy.core.multiarray import array as array
import pandas as pd
from h5py import File
from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY,
    FieldReaderFactory,
    GridReader
)
from magnet_pinn.preprocessing.reading_properties import PropertyReader
from magnet_pinn.preprocessing.voxelizing_mesh import MeshVoxelizer
from magnet_pinn.preprocessing.simulation import Simulation

RAW_DATA_DIR_PATH = "raw"
ANTENNA_MATERIALS_DIR_PATH = osp.join("antenna", "dipole", "raw")
INPUT_DIR_PATH = "Input"
PROCESSED_DIR_PATH = "processed"
INPUT_SIMULATIONS_DIR_PATH = "simulations"
PROCESSED_SIMULATIONS_DIR_PATH = "simulations"
INPUT_ANTENNA_DIR_PATH = "antenna"
PROCESSED_ANTENNA_DIR_PATH = "antenna"
TARGET_FILE_NAME = "{name}.h5"

STANDARD_VOXEL_SIZE = 4
FEATURE_NAMES = ("conductivity", "permittivity", "density")
AIR_FEATURES = {"conductivity": 0.0, "permittivity": 1.0006, "density": 1.293}
AIR_FEATURE_VALUES = np.array(tuple(AIR_FEATURES[feature_name] for feature_name in FEATURE_NAMES))

COMPLEX_DTYPE_KIND = "c"
FLOAT_DTYPE_KIND = "f"


class Preprocessing(ABC):
    """
    Abstract class for preprocessing. 
    Describes the general structure of the preprocessing pipeline.

    First of all we check input and output directory structures and read antenna 
    data which will be used for the whole batch. The main method `process_simulations` 
    make some calculations and save processed data to the output directory.

    Attributes
    ----------
    field_dtype : np.dtype
        type of saving field data
    simulations_dir_path : str
        Simulations location in the batch directory
    out_simmulations_dir_path : str
        Processed simulations location in the output directory
    out_antenna_dir_path : str
        Processed antenna location in the output directory
    dipoles_properties : pd.DataFrame
        Antenna feature dataframe including dipoles meshes files
    dipoles_meshes : list
        A list of dipoles meshes
    dipoles_features : np.array
        Calculated dipoles features in each measurement point
    dipoles_masks : np.array
        Dipoles mask in each measurement point
    """

    def __init__(self, batch_dir_path: str, output_dir_path: str, field_dtype: np.dtype = np.complex64) -> None:
        """
        Parameters
        ----------
        batch_dir_path : str
            Path to the batch directory
        output_dir_path : str
            Path to the output directory
        """
        self.field_dtype = np.dtype(field_dtype)
        self.simulations_dir_path = osp.join(batch_dir_path, INPUT_SIMULATIONS_DIR_PATH)

        # create output directories
        target_dir_name = self._output_target_dir
        self.out_simmulations_dir_path = osp.join(
            output_dir_path,
            target_dir_name,
            PROCESSED_SIMULATIONS_DIR_PATH
        )
        makedirs(self.out_simmulations_dir_path, exist_ok=True)

        self.out_antenna_dir_path = osp.join(
            output_dir_path,
            target_dir_name,
            PROCESSED_ANTENNA_DIR_PATH
        )
        makedirs(self.out_antenna_dir_path, exist_ok=True)

        self.dipoles_properties, self.dipoles_meshes = self.__get_properties_and_meshes(
            osp.join(batch_dir_path, INPUT_ANTENNA_DIR_PATH)
        )
        self.dipoles_features, self.dipoles_masks = None, None

    @property
    @abstractmethod
    def _output_target_dir(self) -> str:
        """
        Gives the name of the simulations output directory 
        in the batch output directory
        """
        pass

    def __get_properties_and_meshes(self, dir_path: str) -> Tuple:
        """
        Reads properties file `materials.txt` as csv file and then 
        loads meshes files which are mentioned in the dataframe and 
        located in the same directory.
        Parameters
        ----------
        dir_path : str
            Path to the data directory

        Returns
        -------
        Tuple
            A tuple of dataframe with properties and a list of meshes
        """
        property_reader = PropertyReader(dir_path)
        meshes = property_reader.read_meshes()
        return (
            property_reader.properties,
            meshes,
        )

    def process_simulations(self, simulation_names: str | None = None):
        """
        Main processing method. It processes all simulations in the batch
        or that one which are mentioned in the `simulation_names` list.

        This method make iteration over all simulation directories found in the 
        `dir_path` and calls `__process_simulation` method for each of it.
        After the main work is done it also calls `_write_dipoles` method 
        to save processed antenna data. 

        Parameters
        ----------
        simulation_names : str | None
            A list of simulation names which should be processed. 
            If None, all simulations will be processed.
        """

        full_sim_list = listdir(self.simulations_dir_path)

        if simulation_names is None:
            simulation_names = full_sim_list
        elif not set(simulation_names).issubset(full_sim_list):
            raise Exception("Simulations are not valid")

        for simulation_name in tqdm(simulation_names):
            self.__process_simulation(simulation_name)
        
        self._write_dipoles()

    def __process_simulation(self, simulation_name: str):
        """
        The main internal method to make simulation processing.

        It creates a `Simulation` instance and then passes it one by one
        into preprocessing steps, which save data into the instance 
        as properties. After the simulation data is ready it calls 
        `_format_and_write_dataset` method to save the data into the output directory.

        Parameters
        ----------
        simulation_name : str
            Name of the simulation which is also the simulation directory name
        """
        simulation = Simulation(
            name=simulation_name,
            path=osp.join(self.simulations_dir_path, simulation_name),
        )

        self._extract_fields_data(simulation)

        self.__calculate_features(simulation)

        self._format_and_write_dataset(simulation)

    @abstractmethod
    def _extract_fields_data(self, out_simulation: Simulation):
        """
        Extracts field data from the simulation directory 
        and saves it into the `out_simulation` instance.

        Parameters
        ----------
        out_simulation : Simulation
            object where to save a data.
        """
        pass

    def __calculate_features(self, out_simulation: Simulation) -> None:
        """
        Calculates or extracts masks and features for both subject and antenna.

        The method use `__get_properties_and_meshes` method and 
        `_get_objects_features_and_mask` to finally calculate object features. 
        Also it uses the precomputed antenna data from `_get_dipoles_features_and_mask` 
        to calculate the final features and masks for the simulation.

        Parameters
        ----------
        out_simulation : Simulation
            The instance to save the data
        """
        object_properties, object_meshes = self.__get_properties_and_meshes(
            osp.join(out_simulation.path, INPUT_DIR_PATH)
        )

        objects_features, object_masks = self._get_features_and_mask(
            object_properties, object_meshes
        )

        dipoles_features, dipoles_mask = self._get_dipoles_features_and_mask()

        features = dipoles_features + objects_features

        features = self._set_air_features(features)

        generak_mask = self._get_general_mask(dipoles_mask, object_masks)

        out_simulation.features = features
        out_simulation.general_mask = generak_mask
        out_simulation.object_masks = object_masks

    @abstractmethod
    def _set_air_features(self, features: np.array) -> np.array:
        """
        Methods processes air features.
        """
        pass

    def _get_general_mask(self, dipoles_mask: np.array, objects_mask: np.array) -> np.array:
        all_masks, _ = pack(
            (dipoles_mask, objects_mask),
            self._concat_masks_pattern
        )
        general_mask = reduce(
            all_masks,
            self._sum_masks_pattern,
            "sum"
        ).astype(bool)

        return general_mask
    
    @property
    @abstractmethod
    def _concat_masks_pattern():
        pass

    @property
    @abstractmethod
    def _sum_masks_pattern():
        pass

    @abstractmethod
    def _get_features(self, properties: pd.DataFrame, masks:np.array) -> np.array:
        """
        A shortcut for the procedure of multiplication of properties and masks.

        The method manipulates masks and properties shapes to calculate element-wise
        product and sum over the component axis.

        Parameters
        ----------
        properties : pd.DataFrame
            A properties frame
        masks : np.array
            A mask array

        Returns
        -------
        np.array
            Calculated features
        """
        pass
    
    @abstractmethod
    def _get_features_and_mask(self, properties: pd.DataFrame, meshes: List) -> Tuple:
        """
        Calculates features and masks based on given parameters.

        Parameters
        ----------
        properties : pd.DataFrame
            A properties frame
        meshes : List
            A list of meshes

        Returns
        -------
        Tuple
            A tuple of features and masks
        """
        pass

    @abstractmethod
    def _get_dipoles_features_and_mask(self) -> Tuple:
        """
        Calculates features and masks for the antenna.

        Returns
        -------
        Tuple
            A tuple of features and masks
        """
        pass
    
    @abstractmethod
    def _format_and_write_dataset(self, out_simulation: Simulation):
        """
        The final stage for the simulation processing.

        The method formats data from the `out_simulation` instance 
        and writes it to the output directory. 
        """
        pass

    @abstractmethod
    def _write_dipoles(self) -> None:
        """
        Write dipoles masks to the output directory.
        """
        pass

    def _format_fields(self, simulation: Simulation) -> Tuple[np.array, np.array]:
        """
        Formats fields data.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object

        Returns
        -------
        np.array:
            e-fiel data
        np.array:
            h-field data
        """
        if self.field_dtype.kind == COMPLEX_DTYPE_KIND:
            e_field = simulation.e_field.astype(self.field_dtype)
            h_field = simulation.h_field.astype(self.field_dtype)
        elif self.field_dtype.kind == FLOAT_DTYPE_KIND:
            e_field = np.array(
                [simulation.e_field.real, simulation.e_field.imag],
                dtype=[("re", self.field_dtype),("im", self.field_dtype)]
            )
            h_field = np.array(
                [simulation.h_field.real, simulation.h_field.imag],
                dtype=[("re", self.field_dtype),("im", self.field_dtype)]
            )

        return e_field. h_field
            


class GridPreprocessing(Preprocessing):
    """
    Class for preprocessing data for grid-based models.

    The class is responsible for reading and processing antennas and subjects data in a voxel grid manner.

    Attributes
    ----------
    voxel_size : int
        the size of the voxel for creating a grid
    positions_min : np.array
        the minimum values of the extent
    positions_max : np.array
        the maximum values of the extent
    field_dtype : np.dtype
        type of saving field data
    simulations_dir_path : str
        Simulations location in the batch directory
    out_simmulations_dir_path : str
        Processed simulations location in the output directory
    out_antenna_dir_path : str
        Processed antenna location in the output directory
    dipoles_properties : pd.DataFrame
        Antenna feature dataframe including dipoles meshes files
    dipoles_meshes : list
        A list of dipoles meshes
    dipoles_features : np.array
        Calculated dipoles features in each measurement point
    dipoles_masks : np.array
        Dipoles mask in each measurement point
    """
    def __init__(
        self, batch_dir_path: str, 
        output_dir_path: str, 
        voxel_size: int = STANDARD_VOXEL_SIZE, 
        field_dtype: np.dtype = np.complex64, 
        **kwargs
    ):
        """
        It does a standard init, checks the extent, creates a voxelizer and process antenna data.

        Parameters
        ----------
        batch_dir_path : str
            Path to the batch directory
        output_dir_path : str
            Path to the output directory
        voxel_size : int
            The size of the voxel for creating a grid
        """
        self.voxel_size = voxel_size
        super().__init__(batch_dir_path, output_dir_path, field_dtype)

        # check extent for validity
        min_values = np.array(
            (kwargs["x_min"], kwargs["y_min"], kwargs["z_min"]),
            dtype=np.float32
        )
        max_values = np.array(
            (kwargs["x_max"], kwargs["y_max"], kwargs["z_max"]),
            dtype=np.float32
        )
        if not np.all((max_values - min_values) % voxel_size == 0):
            raise Exception("Extent not divisible by voxel size")
        self.positions_min = min_values
        self.positions_max = max_values

        # create a voxelizer
        x_unique = np.arange(min_values[0], max_values[0] + voxel_size, voxel_size)
        y_unique = np.arange(min_values[1], max_values[1] + voxel_size, voxel_size)
        z_unique = np.arange(min_values[2], max_values[2] + voxel_size, voxel_size)
        self.voxelizer = MeshVoxelizer(voxel_size, x_unique, y_unique, z_unique)

        # dipoles features are same for the whole batch, so we can calculate them once
        self.dipoles_features, self.dipoles_masks = self._get_features_and_mask(
            self.dipoles_properties, self.dipoles_meshes
        )

    @property
    def _output_target_dir(self) -> str:
        """
        Gives a name of the simulation out directory based on 
        voxel grid and data type we use to save the field data.
        """
        return f"grid_voxel_size_{self.voxel_size}_data_type_{self.field_dtype.name}"

    def _write_dipoles(self) -> None:
        """
        Writes processed antenna masks to the output directory.
        """
        makedirs(self.out_antenna_dir_path, exist_ok=True)
        
        target_file_name = "antenna.h5"
        with File(osp.join(self.out_antenna_dir_path, target_file_name), "w") as f:
            f.create_dataset("masks", data=self.dipoles_masks)

    def _extract_fields_data(self, out_simulation: Simulation) -> None:
        e_field_reader = FieldReaderFactory(
            out_simulation.path, E_FIELD_DATABASE_KEY
        ).create_reader()
        h_field_reader = FieldReaderFactory(
            out_simulation.path, H_FIELD_DATABASE_KEY
        ).create_reader()

        e_x_bound, e_y_bound, e_z_bound = e_field_reader.coordinates
        h_x_bound, h_y_bound, h_z_bound = h_field_reader.coordinates

        if (
            not np.array_equal(e_x_bound, h_x_bound) 
            or not np.array_equal(e_y_bound, h_y_bound) 
            or not np.array_equal(e_z_bound, h_z_bound)
        ):
            raise Exception("Different coordinate systems for E and H fields")
        
        self.__sanity_check(e_x_bound, e_y_bound, e_z_bound)

        out_simulation.e_field = e_field_reader.extract_data().astype(np.complex64)
        out_simulation.h_field = h_field_reader.extract_data().astype(np.complex64)

    def __sanity_check(self, x_bound: np.array, y_bound: np.array, z_bound: np.array) -> None:
        data_min = np.array(
            (np.min(x_bound), np.min(y_bound), np.min(z_bound)),
            dtype=np.float32
        )
        if not np.array_equal(self.positions_min, data_min):
            raise Exception("Min not satisfied")

        data_max = np.array(
            (np.max(x_bound), np.max(y_bound), np.max(z_bound)),
            dtype=np.float32
        )
        if not np.array_equal(self.positions_max, data_max):
            raise Exception("Max not satisfied")

    def _set_air_features(self, features: np.array) -> np.array:
        air_mask = features == 0

        extneded_air_prop = repeat(
            AIR_FEATURE_VALUES,
            "feature -> feature x y z",
            x=features.shape[1],
            y=features.shape[2],
            z=features.shape[3]
        )

        return features + extneded_air_prop * air_mask
        
    def _get_features(self, properties: pd.DataFrame, masks:np.array) -> np.array:
        """
        A shortcut for the procedure of multiplication of properties and masks.

        The method manipulates masks and properties shapes to calculate element-wise
        product and sum over the component axis.

        Parameters
        ----------
        properties : pd.DataFrame
            A properties frame
        masks : np.array
            A mask array

        Returns
        -------
        np.array
            Calculated features
        """
        
        props = properties.loc[:, FEATURE_NAMES].to_numpy().T
        extended_props = repeat(
            props,
            "feature component -> feature x y z component",
            x=masks.shape[0],
            y=masks.shape[1],
            z=masks.shape[2]
        )

        extended_masks = repeat(
            masks,
            "x y z component -> feature x y z component",
            feature=len(FEATURE_NAMES)
        )

        result = reduce(
            extended_props * extended_masks,
            "feature x y z component -> feature x y z",
            "sum"
        )

        return result
        
    def _get_features_and_mask(self, properties, meshes) -> Tuple:
        mask = rearrange(
            list(map(
                lambda x: self.voxelizer.process_mesh(x),
                meshes,
            )),
            "component x y z -> x y z component"
        )
        
        features = self._get_features(properties, mask)
        return (
            features,
            mask
        ) 
        
    def _get_dipoles_features_and_mask(self) -> Tuple:
        return (
            self.dipoles_features,
            self.dipoles_masks
        )
    
    @property
    def _concat_masks_pattern():
        return "x y z *"

    @property
    def _sum_masks_pattern():
        return "x y z component -> x y z"

    def _format_and_write_dataset(self, out_simulation: Simulation) -> None:
        makedirs(self.out_simmulations_dir_path, exist_ok=True)

        output_file_path = osp.join(
            self.out_simmulations_dir_path,
            TARGET_FILE_NAME.format(name=out_simulation.name)
        )

        e_field, h_field = self._format_fields(out_simulation)

        with File(output_file_path, "w") as f:
            f.create_dataset("input", data=out_simulation.features)
            f.create_dataset("efield", data=e_field)
            f.create_dataset("hfield", data=h_field)
            f.create_dataset("subject", data=out_simulation.object_masks)
            f.create_dataset("general_mask", data=out_simulation.general_mask)


class GraphPreprocessing(Preprocessing):

    _coordinates = None

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        if self._coordinates is None:
            self._coordinates = coordinates
        else:
            if not np.array_equal(self._coordinates, coordinates):
                raise Exception("Different coordinate systems for simulations")

    @property
    def _output_target_dir(self) -> str:
        """
        Names the out simulation directory.
        """
        return f"graph_data_type_{self.field_dtype.name}"
    
    def _extract_fields_data(self, out_simulation: Simulation) -> None:
        e_field_reader = FieldReaderFactory(
            out_simulation.path, E_FIELD_DATABASE_KEY
        ).create_reader()
        h_field_reader = FieldReaderFactory(
            out_simulation.path, H_FIELD_DATABASE_KEY
        ).create_reader()

        if isinstance(e_field_reader, GridReader):
            e_field_reader.as_grid = False
            h_field_reader.as_grid = False

        e_coordinates = e_field_reader.coordinates
        h_coordinates = h_field_reader.coordinates

        if (
            not np.array_equal(e_coordinates, h_coordinates)
        ):
            raise Exception("Different coordinate systems for E and H fields")
        
        self.coordinates = e_coordinates

        out_simulation.coordinates = e_coordinates
        out_simulation.e_field = e_field_reader.extract_data().astype(np.complex64)
        out_simulation.h_field = h_field_reader.extract_data().astype(np.complex64)

    def _get_features(self, properties: pd.DataFrame, masks: np.array) -> np.array:
        props = properties.loc[:, FEATURE_NAMES].to_numpy().T
        extended_props = repeat(
            props,
            "feature component -> points feature component",
            points=masks.shape[0]
        )

        extended_masks = repeat(
            masks,
            "points component -> points feature component",
            feature=len(FEATURE_NAMES)
        )

        result = reduce(
            extended_props * extended_masks,
            "points feature component -> points feature",
            "sum"
        )

        return result

    def _set_air_features(self, features: np.array) -> np.array:
        air_mask = features == 0

        extneded_air_prop = repeat(
            AIR_FEATURE_VALUES,
            "feature -> points feature",
            points=features.shape[0]
        )

        return features + extneded_air_prop * air_mask

    def _get_features_and_mask(self, properties, meshes) -> Tuple:
        mask = np.stack(
            list(map(
                lambda x: x.contains(self.coordinates),
                meshes,
            )),
            axis=-1
        )
        features = self._get_features(properties, mask)
        return (
            features,
            mask
        )
    
    @property
    def _concat_masks_pattern():
        return "points *"

    @property
    def _sum_masks_pattern():
        return "points component -> points"
    
    def _get_dipoles_features_and_mask(self) -> Tuple:
        if self.dipoles_features is None or self.dipoles_masks is None:
            self.dipoles_features, self.dipoles_masks = self._get_features_and_mask(
                self.dipoles_properties, self.dipoles_meshes
            )
        
        return (
            self.dipoles_features,
            self.dipoles_masks
        )

    def _format_and_write_dataset(self, out_simulation: Simulation):
        makedirs(self.out_simmulations_dir_path, exist_ok=True)
        output_file_path = osp.join(
            self.out_simmulations_dir_path,
            TARGET_FILE_NAME.format(name=out_simulation.name)
        )

        e_field, h_field = self._format_fields(out_simulation)

        with File(output_file_path, "w") as f:
            f.create_dataset("input", data=out_simulation.features)
            f.create_dataset("efield", data=e_field)
            f.create_dataset("hfield", data=h_field)
            f.create_dataset("subject", data=out_simulation.object_masks)
            f.create_dataset("positions", out_simulation.coordinates)
            f.create_dataset("general_mask", data=out_simulation.general_mask)
