"""Module for reading field values"""

import fnmatch
import os
import os.path as osp

import numpy as np
from h5py import File

E_FIELD_DATABASE_KEY = "E-Field"
H_FIELD_DATABASE_KEY = "H-Field"
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


class FieldReader:
    """Class for reading H/E field values.
    We assume values can be only packed in .h5/.txt files"""

    def __init__(
        self, simulation_dir_path: str, field_type: str = E_FIELD_DATABASE_KEY
    ):
        self.field_type = field_type

        field_dir_path = osp.join(simulation_dir_path, FIELD_DIR_PATH[field_type])
        self.files_list = list(
            map(
                lambda x: osp.join(field_dir_path, x),
                sorted(os.listdir(field_dir_path)),
            )
        )

        if len(self.files_list) == 0:
            raise Exception(
                f"""
                No field values found for the simulation
                {osp.basename(simulation_dir_path)}
                for the {field_type} field
                """
            )

        if len(fnmatch.filter(self.files_list, H5_FILENAME_PATTERN)) != 0:
            self.__coordinates_read_method__ = self.__read_coordinates_from_h5__
            self.__data_read_method__ = self.__read_data_from_h5__
        else:
            self.__ascii_columns_order__ = self.__get_real_columns_order__()
            self.__coordinates_read_method__ = self.__read__coordinates_from_ascii__
            self.__data_read_method__ = self.__read_data_from_ascii__

        self.coordinates = self.__get__cordinates__()

    def read_data(self):
        return np.stack(list(map(self.__data_read_method__, self.files_list)), axis=-1)

    def __read_data_from_h5__(self, file_path: str):
        with File(file_path) as f:
            values = f.get(self.field_type)[:]

        # Extract Ex,Ey,Ez as complex
        Ex = values["x"]["re"] + 1j * values["x"]["im"]
        Ey = values["y"]["re"] + 1j * values["y"]["im"]
        Ez = values["z"]["re"] + 1j * values["z"]["im"]

        return np.column_stack((Ex, Ey, Ez)).astype(np.complex128)

    def __get__cordinates__(self):
        coordinates = self.__coordinates_read_method__(self.files_list[0])
        for file_path in self.files_list[1:]:
            other_coordinatess = self.__coordinates_read_method__(file_path)
            if not np.array_equal(coordinates, other_coordinatess):
                raise Exception(
                    f"Different coordinates in the field value file {file_path}"
                )
        return coordinates

    def __read_coordinates_from_h5__(self, file_path: str):
        with File(file_path) as f:
            positions = f[POSITIONS_DATABASE_KEY][:]

        return np.column_stack((positions["x"], positions["y"], positions["z"])).astype(
            np.int64
        )

    def __get_real_columns_order__(self):
        header = np.loadtxt(self.files_list[0], max_rows=1, dtype=str)
        header = list(header[::2])
        columns_order = ASCII_COLUMN_NAMES[self.field_type]
        return [header.index(col) for col in columns_order]

    def __read__coordinates_from_ascii__(self, file_path: str):
        data = np.loadtxt(file_path, skiprows=2)[:, self.__ascii_columns_order__]
        return data[:, :3].astype(np.int64)

    def __read_data_from_ascii__(self, file_path: str):
        data = np.loadtxt(file_path, skiprows=2)
        data = data[:, self.__ascii_columns_order__]

        Ex = data[:, 3] + 1j * data[:, 6]
        Ey = data[:, 4] + 1j * data[:, 7]
        Ez = data[:, 5] + 1j * data[:, 8]

        return np.column_stack((Ex, Ey, Ez)).astype(np.complex128)
