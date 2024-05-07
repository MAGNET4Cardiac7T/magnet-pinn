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

        if len(fnmatch.filter(self.files_list, H5_FILENAME_PATTERN)) != 0:
            self.read_data = self.__read_all_h5_files
        else:
            self.ascii_columns_order = self.__get_real_columns_order()
            self.read_data = self.__read_all_ascii_files

    def __read_all_h5_files(self):
        return map(lambda x: self.__read_from_h5(x), self.files_list)

    def __read_from_h5(self, file_path: str):

        with File(file_path) as f:
            values = f.get(self.field_type)[:]
            positions = f[POSITIONS_DATABASE_KEY][:]

        # Extract Ex,Ey,Ez as complex
        Ex = values["x"]["re"] + 1j * values["x"]["im"]
        Ey = values["y"]["re"] + 1j * values["y"]["im"]
        Ez = values["z"]["re"] + 1j * values["z"]["im"]

        return np.column_stack((Ex, Ey, Ez)), positions

    def __get_real_columns_order(self):
        header = np.loadtxt(self.files_list[0], max_rows=1, dtype=str)
        header = list(header[::2])
        columns_order = ASCII_COLUMN_NAMES[self.field_type]
        return [header.index(col) for col in columns_order]

    def __read_all_ascii_files(self):
        return map(lambda x: self.__read_from_ascii(x), self.files_list)

    def __read_from_ascii(self, file_path: str):
        data = np.loadtxt(file_path, skiprows=2)
        data = data[:, self.ascii_columns_order]

        Ex = data[:, 3] + 1j * data[:, 6]
        Ey = data[:, 4] + 1j * data[:, 7]
        Ez = data[:, 5] + 1j * data[:, 8]

        positions = data[:, :3].astype(np.int64)
        return np.column_stack((Ex, Ey, Ez)), positions
