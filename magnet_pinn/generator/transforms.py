from abc import ABC, abstractmethod

import igl
import trimesh
from trimesh import Trimesh

from .typing import PhantomItem
from .serializers import MeshSerializer


class Transform(ABC):

    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError("Subclasses must implement `__call__` method")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Compose(Transform):

    def __init__(self, transforms: list[Transform], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __call__(self, tissue: PhantomItem, *args, **kwds) :
        for transform in self.transforms:
            tissue = transform(tissue, *args, **kwds)
        return tissue

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([str(t) for t in self.transforms])})"
    

class ToMesh(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serializer = MeshSerializer()

    def __call__(self, tissue: PhantomItem, *args, **kwds) -> PhantomItem:
        return PhantomItem(
            parent=self.serializer.serialize(tissue.parent),
            children=[self.serializer.serialize(c) for c in tissue.children],
            tubes=[self.serializer.serialize(t) for t in tissue.tubes]
        )


class MeshesCutout(Transform):

    def __call__(self, tissue: PhantomItem, *args, **kwds):
        return PhantomItem(
            parent=self._cut_parent(tissue),
            children=self._cut_children(tissue),
            tubes=self._cut_tubes(tissue)
        )

    def _cut_parent(self, tissue: PhantomItem) -> Trimesh:
        cutters = tissue.children + tissue.tubes
        union_cutters = trimesh.boolean.union(cutters, engine='manifold')
        return trimesh.boolean.difference([tissue.parent, union_cutters], engine='manifold')

    def _cut_children(self, tissue: PhantomItem) -> list[Trimesh]:
        if not tissue.tubes:
            return []
        
        tubes_union = trimesh.boolean.union(tissue.tubes, engine='manifold')
        return [
            trimesh.boolean.difference([c, tubes_union], engine='manifold')
            for c in tissue.children
        ]
    
    def _cut_tubes(self, tissue: PhantomItem) -> list[Trimesh]:
        return [
            trimesh.boolean.intersection([t, tissue.parent], engine='manifold')
            for t in tissue.tubes
        ]
    

class MeshesCleaning(Transform):
    def __call__(self, tissue: PhantomItem, *args, **kwds):
        return PhantomItem(
            parent=self._clean_mesh(tissue.parent),
            children=[self._clean_mesh(c) for c in tissue.children],
            tubes=[self._clean_mesh(t) for t in tissue.tubes]
        )
    
    def _clean_mesh(self, mesh: Trimesh) -> Trimesh:
        mesh = mesh.copy()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        trimesh.repair.fill_holes(mesh)
        mesh.merge_vertices()
        mesh.fix_normals()
        mesh.remove_unreferenced_vertices()
        return mesh
    

class MeshesRemesh(Transform):

    def __init__(self, max_len: float = 8.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    def __call__(self, tissue: PhantomItem, *args, **kwds):
        return PhantomItem(
            parent=self._remesh(tissue.parent),
            children=[self._remesh(c) for c in tissue.children],
            tubes=[self._remesh(t) for t in tissue.tubes]
        )
    
    def _remesh(self, mesh: Trimesh) -> Trimesh:
        v, f = trimesh.remesh.subdivide_to_size(
            mesh.vertices, 
            mesh.faces, 
            max_edge=self.max_len
        )
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        return mesh
