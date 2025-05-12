import os
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from trimesh import Trimesh

# implement a .stl support files and support extension for .step easily


class Serializer(ABC):
    def __init__(self, properties_cfg, dirname: str, seed: int = 42):
        self.properties_cfg = properties_cfg
        self.dirname = dirname
        self.seed = seed
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    @abstractmethod
    def serialize(self, mesh: Trimesh, filepath: str):
        pass

    @abstractmethod
    def get_filename(self, mesh_type: str, index: int = None) -> str:
        pass

    def export(self, meshes):
        parent, children, tubes = meshes
        logging.info(f"Exporting meshes to {self.dirname}.")
        materials = []
        np.random.seed(self.seed)

        logging.info("Exporting parent mesh.")
        props = self._generate_properties()
        fname = self.get_filename("parent")
        props["file"] = fname
        materials.append(props)
        self.serialize(parent, os.path.join(self.dirname, fname))

        logging.info("Exporting children meshes.")
        for i, mesh in enumerate(children):
            props = self._generate_properties()
            fname = self.get_filename("child", i)
            props["file"] = fname
            materials.append(props)
            self.serialize(mesh, os.path.join(self.dirname, fname))

        logging.info("Exporting tube meshes.")
        for i, mesh in enumerate(tubes):
            props = self._generate_properties()
            fname = self.get_filename("tube", i)
            props["file"] = fname
            materials.append(props)
            self.serialize(mesh, os.path.join(self.dirname, fname))

        logging.info("Exporting materials table.")
        df = pd.DataFrame(materials)
        df = df[["file", "conductivity", "permittivity", "density"]]
        df.to_csv(os.path.join(self.dirname, "materials.txt"), index=False)

    def _generate_properties(self):
        props = {}
        for key, val in self.properties_cfg.items():
            props[key] = np.random.uniform(val.min, val.max)
        return props


class STLSerializer(Serializer):
    def serialize(self, mesh: Trimesh, filepath: str):
        mesh.export(filepath)

    def get_filename(self, mesh_type: str, index: int = None) -> str:
        if mesh_type == "parent":
            return "parent_blob.stl"
        elif mesh_type == "child":
            return f"child_blob_{index+1}.stl"
        elif mesh_type == "tube":
            return f"tube_{index+1}.stl"
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")
