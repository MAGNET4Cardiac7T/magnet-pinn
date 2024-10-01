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
AIR_FEATURES = {"conductivity": 0.0, "permittivity": 1.0006, "density": 1.293}
FEATURE_NAMES = list(AIR_FEATURES.keys())
FILE_COLUMN_NAME = "file"


class PropertyReader:
    """
    This class is responsible for reading the properties of the materials.

    We assumed directory mentioned in the `properties_dir_path` has such a structure:

    | ./properties_dir_path
    |    ├── materials.txt
    |    ├── *.stl

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
        materials_file = osp.join(self.properties_dir_path, MATERIALS_FILE_NAME)
        if not osp.exists(materials_file):
            raise FileNotFoundError(f"File {materials_file} not found")
        
        prop = pd.read_csv(materials_file)
        if not set([FILE_COLUMN_NAME] + FEATURE_NAMES).issubset(prop.columns):
            raise ValueError(f"File {materials_file} does not have the required columns")
        self.properties = prop

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
        return self.properties[FILE_COLUMN_NAME].apply(self._load_mesh).tolist()
    
    def _load_mesh(self, file_name) -> Trimesh:
        """
        Loads a mesh from the file

        Parameters
        ----------
        file_name: str
            a file name of the mesh

        Returns
        -------
        Trimesh
            Mesh object
        """
        file_path = osp.join(self.properties_dir_path, file_name)
        if not osp.exists(osp.join(file_path)):
            raise FileNotFoundError(f"Mesh file {file_path} not found")
        
        return load_mesh(file_path)
