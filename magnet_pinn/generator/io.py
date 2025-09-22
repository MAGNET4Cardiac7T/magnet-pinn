"""
NAME
    io.py

DESCRIPTION
    This module provides file I/O functionality for mesh phantoms and their properties.
    It contains writers for exporting mesh data and associated material properties to 
    standard file formats used in MRI simulation workflows.
"""
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
from trimesh import Trimesh

from .typing import PropertyPhantom, PropertyItem, MeshPhantom


PARENT_BLOB_FILE_NAME = "parent.stl"
CHILD_BLOB_FILE_NAME = "child_blob_{i}.stl"
TUBE_FILE_NAME = "tube_{i}.stl"
MATERIALS_FILE_NAME = "materials.txt"


class Writer(ABC):
    """
    Abstract base class for writing phantom data to persistent storage.
    
    Provides the common interface for all phantom data writers, ensuring consistent
    output directory management and write operations across different output formats.
    """

    def __init__(self, output_dir: str | Path = Path("data/raw/tissue_meshes"), *args, **kwargs):
        """
        Initialize writer with output directory.

        Parameters
        ----------
        output_dir : str or Path, optional
            Output directory path for written files. Default is "data/raw/tissue_meshes".
            Directory will be created if it doesn't exist.
        *args, **kwargs
            Additional arguments passed to parent class.
        """
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write(self, item: MeshPhantom):
        """
        Write phantom data to persistent storage.

        Parameters
        ----------
        item : MeshPhantom
            The phantom data to write.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class MeshWriter(Writer):
    """
    Writer for exporting mesh phantoms and their material properties.
    
    Exports 3D mesh data as STL files and creates a corresponding materials CSV file
    containing physical properties for each mesh component. The output follows the
    standard format expected by MRI simulation software.
    """
    def write(self, item: MeshPhantom, prop: PropertyPhantom):
        """
        Write mesh phantom and properties to files.
        
        Exports all mesh components as individual STL files and creates a
        corresponding materials.txt CSV file that maps each mesh file to its
        material properties. The output follows the standard format expected
        by MRI simulation software, with systematic naming conventions for
        parent blobs, child blobs, and tubes.

        Parameters
        ----------
        item : MeshPhantom
            The mesh phantom containing parent, children, and tube meshes.
        prop : PropertyPhantom
            The corresponding material properties for each mesh component.
        """

        materials_table = []

        materials_table.append(
            self._save_mesh(item.parent, prop.parent, PARENT_BLOB_FILE_NAME)
        )

        materials_table.extend(
            self._save_mesh(mesh, prop, CHILD_BLOB_FILE_NAME.format(i=i+1))
            for i, (mesh, prop) in enumerate(zip(item.children, prop.children))
        )

        materials_table.extend(
            self._save_mesh(mesh, prop, TUBE_FILE_NAME.format(i=i+1))
            for i, (mesh, prop) in enumerate(zip(item.tubes, prop.tubes))
        )

        df = pd.DataFrame(materials_table)
        df.to_csv(self.dir / MATERIALS_FILE_NAME, index=False)

    def _save_mesh(self, mesh: Trimesh, prop: PropertyItem, filename: str):
        """
        Save a mesh as STL file and return its properties with filename.
        
        Exports the mesh to the specified STL file and augments the property
        dictionary with the filename for later use in the materials dataframe.

        Parameters
        ----------
        mesh : Trimesh
            The triangular mesh to export as STL.
        prop : PropertyItem
            Material properties for this mesh component.
        filename : str
            Name of the STL file to create.

        Returns
        -------
        dict
            Property dictionary with added 'file' key containing the filename.
        """
        file_path = self.dir / filename
        mesh.export(file_path)

        prop_dict = prop.__dict__.copy()
        prop_dict["file"] = filename
        
        return prop_dict
