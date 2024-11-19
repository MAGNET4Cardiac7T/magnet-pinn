"""
NAME
    preprocessing.py

DESCRIPTION
    This module contains preprocessing classes for the simulation data.

CLASSES
    Preprocessing
    GridPreprocessing
    PointPreprocessing
"""
import os.path as osp
from os import makedirs, listdir
from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
from h5py import File
from tqdm import tqdm
from trimesh import Trimesh
from einops import rearrange, repeat, reduce
from igl import fast_winding_number_for_meshes

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    H_FIELD_DATABASE_KEY,
    FieldReaderFactory,
    FieldReader
)
from magnet_pinn.preprocessing.simulation import Simulation
from magnet_pinn.preprocessing.voxelizing_mesh import MeshVoxelizer
from magnet_pinn.preprocessing.reading_properties import PropertyReader
from magnet_pinn.preprocessing.reading_properties import FEATURE_NAMES, AIR_FEATURES

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
AIR_FEATURE_VALUES = np.array(tuple(AIR_FEATURES.values()), dtype=np.float32)

COMPLEX_DTYPE_KIND = "c"
FLOAT_DTYPE_KIND = "f"

FEATURES_OUT_KEY = "input"
E_FIELD_OUT_KEY = "efield"
H_FIELD_OUT_KEY = "hfield"
SUBJECT_OUT_KEY = "subject"
COORDINATES_OUT_KEY = "positions"
MIN_EXTENT_OUT_KEY = "min_extent"
MAX_EXTENT_OUT_KEY = "max_extent"
VOXEL_SIZE_OUT_KEY = "voxel_size"
ANTENNA_MASKS_OUT_KEY = "masks"
DTYPE_OUT_KEY = "dtype"
TRUNCATION_COEFFICIENTS_OUT_KEY = "truncation_coefficients"


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
    _dipoles_features : np.array
        Calculated dipoles features in each measurement point
    _dipoles_masks : np.array
        Dipoles mask in each measurement point

    Methods
    -------
    __init__(batch_dir_path: str, output_dir_path: str, field_dtype: np.dtype = np.complex64)
        Initializes the preprocessing object
    process_simulations(simulation_names: Union[List[str], None] = None)
        Main processing method. It processes all simulations in the batch
    """

    __dipoles_masks = None
    __dipoles_features = None

    @property
    def _dipoles_masks(self) -> np.array:
        """
        A getter for the dipoles masks, caclulates it when the user first needs it together with features.
        """
        if self.__dipoles_masks is None:
            self.__dipoles_masks, self.__dipoles_features = self._get_features_and_mask(
                self.dipoles_properties, self.dipoles_meshes
            )
        return self.__dipoles_masks
    
    @property
    def _dipoles_features(self) -> np.array:
        """
        A getter for the dipoles features, caclulates it when the user first needs it together with masks.
        """
        if self.__dipoles_features is None:
            self.__dipoles_features, self.__dipoles_masks = self._get_features_and_mask(
                self.dipoles_properties, self.dipoles_meshes
            )
        return self.__dipoles_features

    def __init__(self, batch_dir_path: str, output_dir_path: str, field_dtype: np.dtype = np.complex64) -> None:
        """
        Parameters
        ----------
        batch_dir_path : str
            Path to the batch directory
        output_dir_path : str
            Path to the output directory
        field_dtype : np.dtype
            type of saving field data
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

    @property
    @abstractmethod
    def _output_target_dir(self) -> str:
        """
        Gives the name of the simulations output directory 
        in the batch output directory
        """
        pass

    def __get_properties_and_meshes(self, dir_path: str) -> Tuple[pd.DataFrame, List[Trimesh]]:
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
        pd.DataFrame
            a dataframe with properties
        List[Trimesh]
            a list of meshes
        """
        property_reader = PropertyReader(dir_path)
        meshes = property_reader.read_meshes()
        return (
            property_reader.properties,
            meshes,
        )

    def process_simulations(self, simulation_names: Union[List[str], None] = None):
        """
        Main processing method. It processes all simulations in the batch
        or that one which are mentioned in the `simulation_names` list.

        This method make iteration over all simulation directories found in the 
        `dir_path` and calls `__process_simulation` method for each of it.
        After the main work is done it also calls `_write_dipoles` method 
        to save processed antenna data. 

        Parameters
        ----------
        simulation_names : List[str] | None
            A list of simulation names which should be processed. 
            If None, all simulations will be processed.
        """

        full_sim_list = listdir(self.simulations_dir_path)

        if simulation_names is not None and not set(simulation_names).issubset(set(full_sim_list)):
            raise Exception("Simulations are not in the directory")
        elif simulation_names is None:
            simulation_names = full_sim_list
        elif not set(simulation_names).issubset(full_sim_list):
            raise Exception("Simulations are not valid")

        pbar = tqdm(simulation_names, total=len(simulation_names))
        for simulation_name in pbar:
            self.__process_simulation(simulation_name)
            pbar.set_postfix({"done": simulation_name}, refresh=True)
        
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

    def _extract_fields_data(self, out_simulation: Simulation):
        """
        Extracts field data from the simulation directory 
        and saves it into the `out_simulation` instance.

        Parameters
        ----------
        out_simulation : Simulation
            object where to save a data.
        """
        e_field_reader = FieldReaderFactory(
            out_simulation.path, E_FIELD_DATABASE_KEY
        ).create_reader(not isinstance(self, PointPreprocessing))
        h_field_reader = FieldReaderFactory(
            out_simulation.path, H_FIELD_DATABASE_KEY
        ).create_reader(not isinstance(self, PointPreprocessing))
        
        self._check_coordinates(e_field_reader, h_field_reader)

        out_simulation.e_field = e_field_reader.extract_data()
        out_simulation.h_field = h_field_reader.extract_data()

    @abstractmethod
    def _check_coordinates(self, e_reader: FieldReader, h_reader: FieldReader) -> None:
        """
        Checks the coordinates of the fields readers.

        Parameters
        ----------
        e_reader : FieldReader
            E-field reader
        h_reader : FieldReader
            H-field reader
        """
        pass

    def __calculate_features(self, out_simulation: Simulation) -> None:
        """
        Calculates or extracts masks and features for both subject and antenna.

        The method use `__get_properties_and_meshes` method and 
        `_get_objects_features_and_mask` to finally calculate object features. 
        Also it uses the precomputed antenna data from to calculate the final 
        features and masks for the simulation.

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

        features = self._dipoles_features + objects_features

        features = self._set_air_features(features)

        out_simulation.features = features
        out_simulation.object_masks = object_masks

    @abstractmethod
    def _set_air_features(self, features: np.array) -> np.array:
        """
        Methods processes air features.
        """
        pass
    
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
        mask = np.ascontiguousarray(rearrange(
            list(map(
                self._get_mask,
                meshes,
            )),
            self._masks_stack_pattern
        ))
        
        features = self._get_features(properties, mask)
        return (
            features,
            mask
        )
    
    @abstractmethod
    def _get_mask(self, mesh: Trimesh) -> np.array:
        """
        A method how to get a mask from the mesh.
        """
        pass

    @property
    @abstractmethod
    def _masks_stack_pattern(self) -> str:
        """
        An einops pattern how to stack masks arrys axis.
        """
        pass

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
        extended_props = self._extend_props(props, masks)

        extended_masks = np.ascontiguousarray(repeat(
            masks,
            self._extend_masks_pattern,
            feature=len(FEATURE_NAMES)
        ))

        result = np.ascontiguousarray(reduce(
            extended_props * extended_masks,
            self._features_sum_pattern,
            "sum"
        ), dtype=np.float32)
        return result

    @abstractmethod
    def _extend_props(self, props: np.array, masks: np.array) -> np.array:
        """
        Calculation of features need to extend properties array to the same shape as masks.

        Parameters
        ----------
        props : np.array
            a properties array
        masks : np.array
            a mask array

        Returns
        -------
        np.array
            an extended properties array
        """
        pass

    @property
    @abstractmethod
    def _extend_masks_pattern(self) -> str:
        """
        An einops pattern how to shape masks to the expected resulting shape.
        """
        pass

    @property
    @abstractmethod
    def _features_sum_pattern(self) -> str:
        """
        An einops pattern how to sum features over the component axis.
        """
        pass
    
    def _format_and_write_dataset(self, out_simulation: Simulation):
        """
        The final stage for the simulation processing.

        The method formats data from the `out_simulation` instance 
        and writes it to the output directory.

        Parameters
        ----------
        out_simulation : Simulation
            The instance with the processed data
        """
        makedirs(self.out_simmulations_dir_path, exist_ok=True)

        output_file_path = osp.join(
            self.out_simmulations_dir_path,
            TARGET_FILE_NAME.format(name=out_simulation.name)
        )

        e_field, h_field = self._format_fields(out_simulation)
        self._feature_truncate_coefficients(out_simulation)
        object_masks = self._format_masks(out_simulation)
        features = self._format_features(out_simulation)

        with File(output_file_path, "w") as f:
            f.create_dataset(FEATURES_OUT_KEY, data=features)
            f.create_dataset(E_FIELD_OUT_KEY, data=e_field)
            f.create_dataset(H_FIELD_OUT_KEY, data=h_field)
            f.create_dataset(SUBJECT_OUT_KEY, data=object_masks)
            self._write_extra_data(out_simulation, f)

    def _feature_truncate_coefficients(self, simulation: Simulation) -> np.array:
        """
            Calculates the coefficients for truncating the features based on the given feature type, to prevent overflow 
            when saving the data as float16.
        """
        if self.field_dtype.name == "float16":
            return np.array([1e+4, 1, 1], dtype=np.float32)
        return np.array([1, 1, 1], dtype=np.float32)

    def _format_masks(self, simulation: Simulation) -> np.array:
        """
        Formats masks data.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object

        Returns
        -------
        np.array:
            masks data
        """

        return simulation.object_masks.astype(np.bool_)
    
    @abstractmethod
    def _format_features(self, simulation: Simulation) -> np.array:
        """
        Formats features data.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object

        Returns
        -------
        np.array:
            features data
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
            e-field data
        np.array:
            h-field data
        """
        e_field, h_field = None, None
        if self.field_dtype.kind == COMPLEX_DTYPE_KIND:
            e_field = simulation.e_field.astype(self.field_dtype)
            h_field = simulation.h_field.astype(self.field_dtype)
        elif self.field_dtype.kind == FLOAT_DTYPE_KIND:
            e_field = np.empty_like(simulation.e_field,
                                    dtype=[("re", self.field_dtype),("im", self.field_dtype)])
            e_field["re"] = simulation.e_field.real
            e_field["im"] = simulation.e_field.imag
            
            h_field = np.empty_like(simulation.h_field,
                                    dtype=[("re", self.field_dtype),("im", self.field_dtype)])
            h_field["re"] = simulation.h_field.real
            h_field["im"] = simulation.h_field.imag
        else:
            raise Exception("Unsupported field data type")

        return e_field, h_field
    
    def _write_extra_data(self, simulation: Simulation, f: File):
        """
        Writes extra data to the output .h5 file.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object
        f : File
            a h5 file descriptor
        """
        f.attrs[DTYPE_OUT_KEY] = self.field_dtype.name
        f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY] = self._feature_truncate_coefficients(simulation)


    def _write_dipoles(self) -> None:
        """
        Write dipoles masks to the output directory.
        """
        makedirs(self.out_antenna_dir_path, exist_ok=True)
        
        target_file_name = TARGET_FILE_NAME.format(name="antenna")
        with File(osp.join(self.out_antenna_dir_path, target_file_name), "w") as f:
            f.create_dataset(ANTENNA_MASKS_OUT_KEY, data=self._dipoles_masks.astype(np.bool_))

            

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
    _dipoles_features : np.array
        Calculated dipoles features in each measurement point
    _dipoles_masks : np.array
        Dipoles mask in each measurement point

    Methods
    -------
    __init__(batch_dir_path: str, output_dir_path: str, voxel_size: int = STANDARD_VOXEL_SIZE, field_dtype: np.dtype = np.complex64)
        Initializes the grid preprocessing object.
    process_simulations(simulation_names: Union[List[str], None] = None)
        Main processing method. It processes all simulations in the batch
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
        field_dtype : np.dtype
            type of saving field data
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

    @property
    def _output_target_dir(self) -> str:
        """
        Gives a name of the simulation out directory based on 
        voxel grid and data type we use to save the field data.
        """
        return f"grid_voxel_size_{self.voxel_size}_data_type_{self.field_dtype.name}"

    def _check_coordinates(self, e_reader: FieldReader, h_reader: FieldReader) -> None:
        """
        Checks the coordinates of the fields

        Parameters
        ----------
        e_reader : FieldReader
            E-field reader
        h_reader : FieldReader
            H-field reader
        """
        x_bound, y_bound, z_bound = e_reader.coordinates
        h_x_bound, h_y_bound, h_z_bound = h_reader.coordinates

        if (
            not np.array_equal(x_bound, h_x_bound) 
            or not np.array_equal(y_bound, h_y_bound) 
            or not np.array_equal(z_bound, h_z_bound)
        ):
            raise Exception("Different coordinate systems for E and H fields")
        
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
        """
        A method checks which voxels are in the air and sets air features to them.

        Parameters
        ----------
        features : np.array
            a precalculated features array
        
        Returns
        -------
        np.array
            a features array
        """
        air_mask = features == 0

        extneded_air_prop = np.ascontiguousarray(repeat(
            AIR_FEATURE_VALUES,
            "feature -> feature x y z",
            x=features.shape[1],
            y=features.shape[2],
            z=features.shape[3]
        ))

        return features + extneded_air_prop * air_mask

    def _get_mask(self, mesh: Trimesh) -> np.array:
        """
        A method returns mask for the mesh.

        For the grid preprocessing we use a voxelizer to get a mask.

        Parameters
        ----------
        mesh : Trimesh
            a mesh object
        
        Returns
        -------
        np.array
            a mask array
        """
        return self.voxelizer.process_mesh(mesh)
    
    @property
    def _masks_stack_pattern(self) -> str:
        """
        An einops pattern how to stack masks arrys axis.

        We store 3D grid first and the last axis is a component axis.

        Returns
        -------
        str
            an einops pattern
        """
        return "component x y z -> x y z component"
    
    def _extend_props(self, props: np.array, masks: np.array) -> np.array:
        """
        Extends properties array to the same shape as masks.

        In the grid preprocessing we keep features axis first, then 3D grid and the last axis is a component axis.

        Parameters
        ----------
        props : np.array
            a properties array
        masks : np.array
            a mask array
        """
        return np.ascontiguousarray(repeat(
            props,
            "feature component -> feature x y z component",
            x=masks.shape[0],
            y=masks.shape[1],
            z=masks.shape[2]
        ))
    
    @property
    def _extend_masks_pattern(self) -> str:
        """
        An einops pattern how to shape masks to the expected resulting shape.
        """
        return "x y z component -> feature x y z component"
    
    @property
    def _features_sum_pattern(self) -> str:
        """
        An einops pattern how to sum features over the component axis.
        """
        return "feature x y z component -> feature x y z"
    
    def _format_features(self, simulation: Simulation) -> np.array:
        """
        Formats features data.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object

        Returns
        -------
        np.array:
            features data
        """
        truncation_coefficients = self._feature_truncate_coefficients(simulation)
        truncation_coefficients = np.expand_dims(truncation_coefficients, axis=(1, 2, 3))
        features = simulation.features / truncation_coefficients
        return features.astype(self.field_dtype.type(0).real)
    
    def _write_extra_data(self, simulation: Simulation, f: File):
        """
        Writes extra data to the output .h5 file.

        Grid preprocessing needs to save voxel size and coordinates
        extent as metadata.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object
        f: h5py.File:
            a h5 file descriptor
        """
        super()._write_extra_data(simulation, f)
        f.attrs[VOXEL_SIZE_OUT_KEY] = self.voxel_size
        f.attrs[MIN_EXTENT_OUT_KEY] = self.positions_min
        f.attrs[MAX_EXTENT_OUT_KEY] = self.positions_max


class PointPreprocessing(Preprocessing):
    """
    Class for preprocessing data for point cloud-based models.

    The class is responsible for reading and processing antennas and subjects data in a point cloud manner.

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
    _dipoles_features : np.array
        Calculated dipoles features in each measurement point
    _dipoles_masks : np.array
        Dipoles mask in each measurement point

    Methods
    -------
    __init__(batch_dir_path: str, output_dir_path: str, field_dtype: np.dtype = np.complex64)
        Initializes the point preprocessing object.
    process_simulations(simulation_names: Union[List[str], None] = None)
        Main processing method. It processes all simulations in the batch
    """
    _coordinates = None

    @property
    def coordinates(self):
        """
        Getter for the coordinates
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        """
        For point preprocessing there are no extent so we check each simulation coordinates if they are the same.
        """
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
        return f"point_data_type_{self.field_dtype.name}"

    def _check_coordinates(self, e_reader: FieldReader, h_reader: FieldReader) -> None:
        """
        Checks the coordinates of the fields

        Parameters
        ----------
        e_reader : FieldReader
            E-field reader
        h_reader : FieldReader
            H-field reader
        """
        e_coordinates = e_reader.coordinates
        h_coordinates = h_reader.coordinates

        if (
            not np.array_equal(e_coordinates, h_coordinates)
        ):
            raise Exception("Different coordinate systems for E and H fields")
        
        self.coordinates = e_coordinates

    def _extend_props(self, props: np.array, masks: np.array) -> np.array:
        """
        Extends properties array to the same shape as masks.

        In point preprocessing we store first the point axis, then property axis and then component axis.

        Parameters
        ----------
        props : np.array
            a properties array
        masks : np.array
            a mask array

        Returns
        -------
        np.array
            an extended properties array
        """
        return np.ascontiguousarray(repeat(
            props,
            "feature component -> points feature component",
            points=masks.shape[0]
        ))
    
    @property
    def _extend_masks_pattern(self) -> str:
        """
        An einops pattern how to shape masks to the expected resulting shape.
        """
        return "points component -> points feature component"
    
    @property
    def _features_sum_pattern(self) -> str:
        """
        An einops pattern how to sum features over the component axis.
        """
        return "points feature component -> points feature"

    def _set_air_features(self, features: np.array) -> np.array:
        """
        A method checks which points are in the air and sets air features to them.

        Parameters
        ----------
        features : np.array
            a precalculated features array

        Returns
        -------
        np.array
            a features array
        """
        air_mask = features == 0

        extneded_air_prop = np.ascontiguousarray(repeat(
            AIR_FEATURE_VALUES,
            "feature -> points feature",
            points=features.shape[0]
        ))

        return features + extneded_air_prop * air_mask
    
    def _get_mask(self, mesh: Trimesh) -> np.array:
        """
        A method returns mask for the mesh.

        In point preprocessing we check each point if it is inside the mesh.
        That is why we calculate the fast winding number and set a threshold 
        to check if is closer to 0 or 1.

        Parameters
        ----------
        mesh : Trimesh
            a mesh object

        Returns
        -------
        np.array
            a mask array
        """
        return fast_winding_number_for_meshes(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int32),
            self.coordinates.astype(np.float32)
        ) > 0.5
    
    @property
    def _masks_stack_pattern(self) -> str:
        """
        An einops pattern how to stack masks arrys axis.
        """
        return "component points -> points component"
    
    def _format_features(self, simulation: Simulation) -> np.array:
        """
        Formats features data.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object

        Returns
        -------
        np.array:
            features data
        """
        truncation_coefficients = self._feature_truncate_coefficients(simulation)
        truncation_coefficients = np.expand_dims(truncation_coefficients, axis=0)
        features = simulation.features / truncation_coefficients
        return features.astype(self.field_dtype.type(0).real)
    
    def _write_extra_data(self, simulation: Simulation, f: File) -> None:
        """
        Writes extra data to the output .h5 file.

        Point preprocessing data needs to save coordinates.

        Parameters
        ----------
        simulation : Simulation
            a simulation data object
        f: h5py.File:
            a h5 file descriptor
        """
        super()._write_extra_data(simulation, f)
        f.create_dataset(COORDINATES_OUT_KEY, data=self.coordinates.astype(np.float32))
