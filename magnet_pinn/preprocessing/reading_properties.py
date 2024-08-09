"""
NAME
    reading_properties.py

DESCRIPTION
    This module is responsible for reading the properties of the materials

CLASSES
    PropertyReader
"""
import os.path as osp
from typing import List

import pandas as pd
from trimesh import load_mesh, Trimesh

MATERIALS_FILE_NAME = "materials.txt"


class PropertyReader:
    """
    This class is responsible for reading the properties of the materials.

    We assumed directory mentioned in the `properties_dir_path` has such a structure:

    | ./properties_dir_path
    |    ├── materials.txt
    |    ├── \*.stl

    Attributes
    ----------
    properties_dir_path : str
        Directory path of the material properties
    properties : pd.DataFrame
        Dataframe containing the properties of the materials

    Methods
    -------
    __init__(properties_dir_path)
        Reads and saves material properties from the directory
    read_meshes()
        Reads the meshes of the materials
    """

    def __init__(self, properties_dir_path: str) -> None:
        """
        Reads and saves material properties from the directory

        Parameters
        ----------
        properties_dir_path : str
            Directory path of the material properties
        """
        self.properties_dir_path = properties_dir_path
        self.properties = pd.read_csv(
            osp.join(self.properties_dir_path, MATERIALS_FILE_NAME)
        )

    def read_meshes(self) -> List[Trimesh]:
        """
        Reads the meshes of the materials

        The `properties` dataframe should have a column named `file` 
        which contains file names of the meshes we need. So here we read the meshes
        and return them as a list.

        Returns
        -------
        List
            List of the meshes of the materials
        """
        return list(map(
                lambda x: load_mesh(osp.join(self.properties_dir_path, x)),
                self.properties["file"].tolist(),
            ))
