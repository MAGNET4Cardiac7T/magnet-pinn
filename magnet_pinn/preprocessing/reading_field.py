"""
NAME
    reading_field.py

DESCRIPTION
    This module contains classes for reading field values from the .h5 files

CLASSES
    FieldReaderFactory
    FieldReader
    GridReader
    PointReader
"""

import os
import fnmatch
import os.path as osp
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from h5py import File
from einops import rearrange

E_FIELD_DATABASE_KEY = "E-Field"
H_FIELD_DATABASE_KEY = "H-Field"
POSITIONS_DATABASE_KEY = "Position"
X_BOUNDS_DATABASE_KEY = "Mesh line x"
Y_BOUNDS_DATABASE_KEY = "Mesh line y"
Z_BOUNDS_DATABASE_KEY = "Mesh line z"

FIELD_DIR_PATH = {E_FIELD_DATABASE_KEY: "E_field", H_FIELD_DATABASE_KEY: "H_field"}
H5_FILENAME_PATTERN = "*AC*.h5"


class FieldReaderFactory:
    """
    Factory class for creating FieldReader objects

    This class checks the type of the files we have in the field directory
    and defines the reader we need

    Parameters
    ----------
    field_type: str
        Type of the field we read and key of the field database in the h5 file.
        Also is used to define the field directory name.
    files_list: list
        The list of files paths with field values which has to be read.

    Assumed Directory structue
    --------------------------
    The simulation directory consists of different field directories

    simulation_dir/

    |- E_field/
    |  |- e-field*(f=*) [AC*]*.h5
    |- H_field/
    |  |- e-field*(f=*) [AC*]*.h5
    |- SAR/
    |  |- SAR*(f=*) [AC*]*.h5

    Methods
    -------
    __init__(simulation_dir_path, field_type)
        It collects the list of source files and check if they exist.
    create_reader(keep_grid_output_format)
        Creates a reader for the field
    """

    def __init__(
        self, simulation_dir_path: str, field_type: str = E_FIELD_DATABASE_KEY
    ):
        """
        It collects the list of source files and check if they exist.

        Parameters
        ----------
        simulation_dir_path: str
            Path to the simulation directory
        field_type: str
            Field type
        
        Raises
        ------
        Exception
            If there are no field values for the simulation
        """

        self.field_type = field_type

        field_dir_path = osp.join(simulation_dir_path, FIELD_DIR_PATH[field_type])
        dir_files = list(
            map(
                lambda x: osp.join(field_dir_path, x),
                sorted(os.listdir(field_dir_path)),
            )
        )
        self.files_list = fnmatch.filter(dir_files, H5_FILENAME_PATTERN)

        if len(self.files_list) == 0:
            raise Exception(
                f"""
                No field values found for the simulation
                {osp.basename(simulation_dir_path)}
                for the {field_type} field
                """
            )

    def create_reader(self, keep_grid_output_format: bool = True):
        """
        Two different types are asumed to be read:

        1. Grid type
            Uses field type as a databae key and also use
            `X_BOUNDS_DATABASE_KEY`, `Y_BOUNDS_DATABASE_KEY`, `Z_BOUNDS_DATABASE_KEY`
            as keys for the coordinates.
        2. Just points
            Field values are under the field type key and the coordinates are under
            `Positions` key.

        Parameters
        ----------
        keep_grid_output_format: bool
            if True, the reader will return the data in the pointslist form

        Returns
        -------
        FieldReader
            prepared reader
        """

        if self.__is_grid():
            instance =  GridReader(self.files_list, self.field_type)
            instance.is_grid = keep_grid_output_format

        else:
            instance = PointReader(self.files_list, self.field_type)
    
        return instance

    def __is_grid(self):
        """
        Checks if the `POSISTIONS_DATABASE_KEY` key is used in h5 file
        """

        with File(self.files_list[0]) as f:
            database_keys = list(f.keys())
        
        return POSITIONS_DATABASE_KEY not in database_keys


class FieldReader(ABC):
    """
    Abstract class for reading field values

    The classs is fully responsible for extracting field values and point coordinates

    Attributes
    ----------
    files_list: list
        The list of files paths which should be read
    field_type: str
        The type of the field we read
    _coordinates: np.ndarray
        The coordinates of the field points

    Methods
    -------
    __init__(files_list, field_type)
        It reads coordinates and validates if the coordinates are the same in all files
    extract_data()
        Extracts field values from the files
    """

    def __init__(self, files_list: list, field_type: str):
        """
        It reads coordinates and makes validation

        Parameters
        ----------
        files_list: list
            The list of files paths which should be processed
        field_type: str
            The type of the field we process
        """

        self.files_list = files_list
        self.field_type = field_type

        self._coordinates = self._read_coordinates(self.files_list[0])
        self.__validate_coordinates()

    def __validate_coordinates(self) -> None:
        """
        It validates coordinates of the field points
        """
        for other_file in self.files_list[1:]:
            other_coordinates = self._read_coordinates(other_file)
            if not self._check_coordinates(other_coordinates):
                raise Exception(
                    f"Different positions in the field value file {other_file}"
                )

    @property        
    @abstractmethod
    def coordinates(self):
        pass

    @abstractmethod
    def _read_coordinates(self, file_path: str) -> Union[Tuple, np.array]:
        """
        Reads coordinates from the h5 file

        Parameters
        ----------
        file_path: str
            The path to the h5 file with field values

        Returns
        -------
        Union[Tuple, np.array]
            The coordinates of the field points
        """

        pass

    @abstractmethod
    def _check_coordinates(self, other_coordinates: Union[Tuple, np.array]) -> bool:
        """
        Checks if the given coordinates are the same as the coordinates saved in the instance

        Parameters
        ----------
        other_coordinates: Union[Tuple, np.array]
            The coordinates to compare with

        Returns
        -------
        bool
            True if the coordinates are the same
        """
        pass

    def extract_data(self) -> np.array:
        """
        Extracts field values from the files

        This is a main method of the class. 
        It reads field values from files and compose it into a single array

        Returns
        -------
        np.array
            The field values
        """
        field_components = list(map(
            self.__read_field_data, self.files_list
        ))

        return self._compose_field_components(field_components)

    def __read_field_data(self, file_path: str) -> np.array:
        """
        Read on field component from the .h5 file

        It reads complex data array and compose it into a single result array

        Parameters
        ----------
        file_path: str
            The path to the h5 file with field values

        Returns
        -------
        np.array
            The field values
        """
        with File(file_path) as f:
            values = f[self.field_type][:]

        Ex = values["x"]["re"] + 1j * values["x"]["im"]
        Ey = values["y"]["re"] + 1j * values["y"]["im"]
        Ez = values["z"]["re"] + 1j * values["z"]["im"]

        return np.ascontiguousarray(rearrange(
            [Ex, Ey, Ez],
            self._compose_field_pattern(Ex.shape)
        ), dtype=np.complex64)

    @abstractmethod
    def _compose_field_pattern(self, data_shape: Tuple) -> str:
        """
        Method returns field array pattern.

        Parameters
        ----------
        data: Tuple
            The field components

        Returns
        -------
        str
            The field array pattern
        """
        pass

    @abstractmethod
    def _compose_field_components(field_components: List) -> np.array:
        """
        Here we compose together field components from different files.

        Parameters
        ----------
        field_components: List
            List of field components

        Returns
        -------
        np.array
            The field values
        """
        pass


class GridReader(FieldReader):
    """
        Class for reading field values in the grid form, i.e the field values are given on the mesh lines

        Attributes
        ----------
        is_grid: bool
            Flag set to true, as the field values are given in the grid form
        files_list: list
            The list of files paths which should be read
        field_type: str
            The type of the field we read
        _coordinates: np.ndarray
            The coordinates of the field points
        
        Methods
        -------
        __init__(files_list, field_type)
            It reads coordinates and validates if the coordinates are the same in all files
        extract_data()
            Extracts field values from the files
        
    """
    is_grid = True

    def _read_coordinates(self, file_path: str) -> Tuple:
        """
        Read coordinates from the h5 file

        In the grid case coordinates are given by the mesh lines. 
        Their access keys are saved in the 
        `X_BOUNDS_DATABASE_KEY`, `Y_BOUNDS_DATABASE_KEY`, `Z_BOUNDS_DATABASE_KEY`.

        Parameters
        ----------
        file_path: str
            The path to the h5 file with field values

        Returns
        -------
        Tuple
            The x, y, z coordinate bounds
        """
        with File(file_path) as f:
            x_bounds = f[X_BOUNDS_DATABASE_KEY][:].astype(np.float64)
            y_bounds = f[Y_BOUNDS_DATABASE_KEY][:].astype(np.float64)
            z_bounds = f[Z_BOUNDS_DATABASE_KEY][:].astype(np.float64)

        return x_bounds, y_bounds, z_bounds
    
    def _check_coordinates(self, other_coordinates: Tuple) -> bool:
        """
        Checks if the given coordinates are the same as the coordinates saved in the instance
        In the grid case we check if all bounds are the same.

        Parameters
        ----------
        other_coordinates: Tuple
            The coordinates to compare with
        """
        x_default_bound, y_default_bound, z_default_bound = self._coordinates
        x_other_bound, y_other_bound, z_other_bound = other_coordinates

        return (
            np.array_equal(x_default_bound, x_other_bound)
            and np.array_equal(y_default_bound, y_other_bound)
            and np.array_equal(z_default_bound, z_other_bound)
            )
    
    @property
    def coordinates(self):
        """
        It suppose just to give back the coordinates list. But if the grid trigger is 
        off, then we give the data back in the pointslist form, that is why we create
        a grid form and reshape it into the form of the pointslist.
        """
        if self.is_grid:
            return self._coordinates
        
        x, y, z = self._coordinates
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        values = np.ascontiguousarray(rearrange(
            [xx, yy, zz],
            "ax x y z -> (x y z) ax"
        ), dtype=np.float32)
        return values
    
    def _compose_field_pattern(self, data_shape: Tuple) -> str:
        """
        The grid data field is an array with 3d grid structure (x, y, z). It checks and fixes axis order.
        """
        expected_shape = {
            "x": self._coordinates[0].shape[0],
            "y": self._coordinates[1].shape[0],
            "z": self._coordinates[2].shape[0]
        }
        if data_shape == tuple(expected_shape.values()):
            axis_order = "x y z"
        elif not np.array_equal(np.sort(data_shape), np.sort(list(expected_shape.values()))):
            raise ValueError("Inconsistent data shape")
        else:
            indices_we_got = np.array(data_shape).argsort()
            indices_we_expect = np.array(list(expected_shape.values())).argsort()
            real_order = np.zeros(3, dtype=int)
            real_order[indices_we_got] = indices_we_expect
            real_order_axis_symbols = np.array(list(expected_shape.keys()))[real_order]
            axis_order = " ".join(real_order_axis_symbols)
        return f"ax {axis_order} -> ax x y z"

    def _compose_field_components(self, field_components: List) -> np.array:
        """
        Compose together field components from different files

        In the grid regime of work we give back the grid data, but if 
        the grid trigger is off, we will give back the data in the pointslist form.

        Parameters
        ----------
        field_components: List
            List of field components

        Returns
        -------
        np.array
            The field values
        """
        if self.is_grid:
            result = rearrange(
                field_components,
                "components ax x y z -> ax x y z components"
            )
        else:
            result = rearrange(
                field_components,
                "components ax x y z -> (x y z) ax components"
            )

        return np.ascontiguousarray(result, dtype=np.complex64)


class PointReader(FieldReader):
    """
    Class for reading field values in the pointslist form, i.e the field values are given at arbitrary point locations.

    Attributes
    ----------
    files_list: list
        The list of files paths which should be read
    field_type: str
        The type of the field we read
    _coordinates: np.ndarray
        The coordinates of the field points

    Methods
    -------
    __init__(files_list, field_type)
        It reads coordinates and validates if the coordinates are the same in all files
    extract_data()
        Extracts field values from the files
    """

    def _read_coordinates(self, file_path: str):
        """
        Read coordinates from the h5 file

        In the pointslist case coordinates have no structure so we just compose them 
        as a list of points.
        """
        with File(file_path) as f:
            x = f[POSITIONS_DATABASE_KEY]["x"][:]
            y = f[POSITIONS_DATABASE_KEY]["y"][:]
            z = f[POSITIONS_DATABASE_KEY]["z"][:]
        return np.ascontiguousarray(rearrange(
            [x, y, z],
            "ax batch -> batch ax"
        ), dtype=np.float32)
    
    def _check_coordinates(self, other_coordinates) -> bool:
        """
        Checks if the given coordinates are the same as the coordinates saved in the instance

        Parameters
        ----------
        other_coordinates: np.array
            The coordinates to compare with

        Returns
        -------
        bool
            True if the coordinates are the same
        """
        return np.array_equal(self._coordinates, other_coordinates)
    
    @property
    def coordinates(self):
        return self._coordinates
    
    @property
    def _compose_field_pattern(self, data_shape: Tuple) -> str:
        """
        This method ignores the existing data because of assumption of 1d
        """
        if data_shape[0] != self._coordinates.shape[0]:
            raise ValueError("Inconsistent data shape")
        
        return "ax batch -> batch ax"
    
    def _compose_field_components(self, field_components: List) -> np.array:
        """
        Compose together field components from different files

        Just return data as a lisst of points measurements.

        Parameters
        ----------
        field_components: List
            List of field components

        Returns
        -------
        np.array
            The field values
        """
        return np.ascontiguousarray(rearrange(
            field_components,
            "components batch ax -> batch ax components"
        ), dtype=np.complex64)
