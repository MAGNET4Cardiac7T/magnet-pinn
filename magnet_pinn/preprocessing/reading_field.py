"""Module for reading field values"""

import fnmatch
import os
import os.path as osp

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


class FieldReader:
    """Class for reading H/E field values.
    We assume values can be only packed in .h5 files"""

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
        
        self.is_grid = self.__is_grid()
        self.positions = self.__extract_positions()
        
    def __is_grid(self):
        with File(self.files_list[0]) as f:
            database_keys = list(f.keys())
        
        return POSISTIONS_DATABASE_KEY not in database_keys
        
    def __extract_positions(self):
        if self.is_grid:
            return self.__get_positions_by_bounds()
        else:
            return self.__get_positions()

    def __get_positions_by_bounds(self):
        x_bound, y_bound, z_bound = self.__read_bounds__(self.files_list[0])

        for file_path in self.files_list[1:]:
            other_x_bound, other_y_bound, other_z_bound = self.__read_bounds__(
                file_path
            )
            if not (
                np.array_equal(x_bound, other_x_bound)
                and np.array_equal(y_bound, other_y_bound)
                and np.array_equal(z_bound, other_z_bound)
            ):
                raise Exception(f"Different bounds in the field value file {file_path}")
            
        xx, yy, zz = np.meshgrid(x_bound, y_bound, z_bound, indexing="ij")
        grid = np.stack((xx, yy, zz), axis=-1)
        positions = grid.reshape(-1, 3)
        return positions

    def __read_bounds__(self, file_path: str):
        with File(file_path) as f:
            x_bounds = f[X_BOUNDS_DATABASE_KEY][:].astype(np.int64)
            y_bounds = f[Y_BOUNDS_DATABASE_KEY][:].astype(np.int64)
            z_bounds = f[Z_BOUNDS_DATABASE_KEY][:].astype(np.int64)

        return x_bounds, y_bounds, z_bounds
    
    def __get_positions(self):
        positions = self.__read_positions(self.files_list[0])
        for file_path in self.files_list[1:]:
            other_positions = self.__read_positions(file_path)
            if not np.array_equal(positions, other_positions):
                raise Exception(f"Different coordinates in the field value file {file_path}")
        
        return positions
    
    def __read_positions(self, file_path: str):
        with File(file_path) as f:
            x = f[POSISTIONS_DATABASE_KEY]["x"][:]
            y = f[POSISTIONS_DATABASE_KEY]["y"][:]
            z = f[POSISTIONS_DATABASE_KEY]["z"][:]
        return np.column_stack((x, y, z)).astype(np.int64)

    def read_data(self):
        field_values = list(map(self.__read_field_data__, self.files_list))
        return np.stack(field_values, axis=-1)

    def __read_field_data__(self, file_path: str):
        with File(file_path) as f:
            values = f[self.field_type][:]

        # Extract Ex,Ey,Ez as complex
        Ex = values["x"]["re"] + 1j * values["x"]["im"]
        Ey = values["y"]["re"] + 1j * values["y"]["im"]
        Ez = values["z"]["re"] + 1j * values["z"]["im"]

        values = np.stack((Ex, Ey, Ez), axis=-1).astype(np.complex128)
        if self.is_grid:
            values = values.reshape(-1, 3)
        
        return values
