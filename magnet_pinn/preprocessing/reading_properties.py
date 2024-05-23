"""Module for reading dipoles/obect properties"""

import os.path as osp

import pandas as pd
from trimesh import load_mesh

MATERIALS_FILE_NAME = "materials.txt"


class PropertyReader:

    def __init__(self, properties_dir_path: str) -> None:
        self.properties_dir_path = properties_dir_path
        self.properties = pd.read_csv(
            osp.join(self.properties_dir_path, MATERIALS_FILE_NAME)
        )

    def read_meshes(self):
        return list(
            map(
                lambda x: load_mesh(osp.join(self.properties_dir_path, x)),
                self.properties["file"].tolist(),
            )
        )
