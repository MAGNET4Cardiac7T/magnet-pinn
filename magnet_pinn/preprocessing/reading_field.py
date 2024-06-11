"""Module for reading field values"""

import os
import fnmatch
import os.path as osp
from abc import ABC, abstractmethod

import numpy as np
from h5py import File

E_FIELD_DATABASE_KEY = "E-Field"
H_FIELD_DATABASE_KEY = "H-Field"
POSISTIONS_DATABASE_KEY = "Position"
X_BOUNDS_DATABASE_KEY = "Mesh line x"
Y_BOUNDS_DATABASE_KEY = "Mesh line y"
Z_BOUNDS_DATABASE_KEY = "Mesh line z"

FIELD_DIR_PATH = {E_FIELD_DATABASE_KEY: "E_field", H_FIELD_DATABASE_KEY: "H_field"}
H5_FILENAME_PATTERN = "*AC*.h5"


class FieldReaderFactory:
    def __init__(
        self, simulation_dir_path: str, field_type: str = E_FIELD_DATABASE_KEY
    ):
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

    def create_reader(self):
        if self.__is_grid():
            return GridReader(self.files_list, self.field_type)
        return PointReader(self.files_list, self.field_type)

    def __is_grid(self):
        with File(self.files_list[0]) as f:
            database_keys = list(f.keys())
        
        return POSISTIONS_DATABASE_KEY not in database_keys


class FieldReader(ABC):
    """Class for reading H/E field values.
    We assume values can be only packed in .h5 files"""

    def __init__(self, files_list: list, field_type: str):
        self.files_list = files_list
        self.field_type = field_type

        self.coordinates = self._read_coordinates(self.files_list[0])
        self.__validate_coordinates()

    def __validate_coordinates(self):
        for other_file in self.files_list[1:]:
            other_coordinates = self._read_coordinates(other_file)
            if not self._check_coordinates(other_coordinates):
                raise Exception(
                    f"Different positions in the field value file {other_file}"
                )

    @abstractmethod
    def _read_coordinates(self, file_path: str):
        pass

    @abstractmethod
    def _check_coordinates(self, other_coordinates):
        pass

    def extract_data(self):
        field_values = list(map(self.__read_field_data, self.files_list))
        return np.stack(field_values, axis=-1)

    def __read_field_data(self, file_path: str):
        with File(file_path) as f:
            values = f[self.field_type][:]

        # Extract Ex,Ey,Ez as complex
        Ex = values["x"]["re"] + 1j * values["x"]["im"]
        Ey = values["y"]["re"] + 1j * values["y"]["im"]
        Ez = values["z"]["re"] + 1j * values["z"]["im"]

        values = np.stack((Ex, Ey, Ez), axis=-1).astype(np.complex128)
        
        return values


class GridReader(FieldReader):
    def _read_coordinates(self, file_path: str):
        with File(file_path) as f:
            x_bounds = f[X_BOUNDS_DATABASE_KEY][:].astype(np.int64)
            y_bounds = f[Y_BOUNDS_DATABASE_KEY][:].astype(np.int64)
            z_bounds = f[Z_BOUNDS_DATABASE_KEY][:].astype(np.int64)

        return x_bounds, y_bounds, z_bounds
    
    def _check_coordinates(self, other_coordinates):
        x_default_bound, y_default_bound, z_default_bound = self.coordinates
        x_other_bound, y_other_bound, z_other_bound = other_coordinates

        return (
            np.array_equal(x_default_bound, x_other_bound)
            and np.array_equal(y_default_bound, y_other_bound)
            and np.array_equal(z_default_bound, z_other_bound)
            )


class PointReader(FieldReader):
    def _read_coordinates(self, file_path: str):
        with File(file_path) as f:
            x = f[POSISTIONS_DATABASE_KEY]["x"][:]
            y = f[POSISTIONS_DATABASE_KEY]["y"][:]
            z = f[POSISTIONS_DATABASE_KEY]["z"][:]
        return np.column_stack((x, y, z)).astype(np.int64)
    
    def _check_coordinates(self, other_coordinates):
        return np.array_equal(self.coordinates, other_coordinates)
