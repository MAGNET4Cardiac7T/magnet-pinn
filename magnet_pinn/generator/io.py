from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
from trimesh import Trimesh

from .typing import PhantomItem, PropertyItem


PARENT_BLOB_FILE_NAME = "parent_blob.stl"
CHILD_BLOB_FILE_NAME = "child_blob_{i}.stl"
TUBE_FILE_NAME = "tube_{i}.stl"
MATERIALS_FILE_NAME = "materials.txt"


class Writer(ABC):

    def __init__(self, dir: str | Path = Path("data/raw/tissue_meshes"), *args, **kwargs):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write(self, item: PhantomItem):
        raise NotImplementedError("Subclasses must implement this method.")


class MeshWriter(Writer):
    def write(self, item: PhantomItem, prop: PhantomItem):

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
        The method gets a mesh, its properties, a directory and a filename.
        It saves a mesh as `.stl` file, saves file name in the one dict of properties and return it.
        P.S. later the general pandas dataframe will be created from these properties.
        """
        file_path = self.dir / filename
        mesh.export(file_path)

        prop = prop.__dict__
        prop["file"] = filename
        
        return prop




