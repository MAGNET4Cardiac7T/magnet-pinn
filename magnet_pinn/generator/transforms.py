from abc import ABC, abstractmethod

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
    

class ToMeshTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serializer = MeshSerializer()

    def __call__(self, tissue: PhantomItem, *args, **kwds) -> PhantomItem:
        return PhantomItem(
            parent=self.serializer.serialize(tissue.parent),
            children=[self.serializer.serialize(c) for c in tissue.children],
            tubes=[self.serializer.serialize(t) for t in tissue.tubes]
        )


class MeshesCutoutTransform(Transform):

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
    

class MeshesCleaningTransform(Transform):
    def __call__(self, tissue: PhantomItem, *args, **kwds):
        return PhantomItem(
            parent=self._clean_mesh(tissue.parent.copy()),
            children=[self._clean_mesh(c.copy()) for c in tissue.children],
            tubes=[self._clean_mesh(t.copy()) for t in tissue.tubes]
        )
    
    def _clean_mesh(self, mesh: Trimesh) -> Trimesh:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        return mesh
    

class MeshesRemeshTransform(Transform):
    def __call__(self, tissue: PhantomItem, max_len: float = 8.0, *args, **kwds):
        return PhantomItem(
            parent=self._remesh(tissue.parent, max_len),
            children=[self._remesh(c, max_len) for c in tissue.children],
            tubes=[self._remesh(t, max_len) for t in tissue.tubes]
        )
    
    def _remesh(self, mesh: Trimesh, max_len: float) -> Trimesh:
        v, f = trimesh.remesh.subdivide_loop(
            mesh.vertices,
            mesh.faces,
            iterations=3
        )
        return trimesh.Trimesh(vertices=v, faces=f)
