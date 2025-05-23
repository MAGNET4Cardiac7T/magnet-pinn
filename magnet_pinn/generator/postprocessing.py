from abc import ABC

import trimesh
from trimesh import Trimesh
from trimesh.remesh import subdivide_to_size

class Postprocessing(ABC):
    def process(self, parent, children, tubes):
        raise NotImplementedError("Subclasses should implement this method.")


class MeshPostprocessing(Postprocessing):
    def process(self, parent: Trimesh, children: list[Trimesh], tubes: list[Trimesh]):
        cutters = children + tubes
        union_cutters = trimesh.boolean.union(cutters, engine='manifold')
        parent_cut = trimesh.boolean.difference([parent, union_cutters], engine='manifold')

        if not tubes:
            children_cut = children.copy()
        else:
            children_cut = []
            tubes_union = trimesh.boolean.union(tubes, engine='manifold')
            for c in children:
                children_cut.append(
                    trimesh.boolean.difference([c, tubes_union], engine='manifold')
                )
         
        tubes_cut = [
            trimesh.boolean.intersection([t, parent], engine='manifold')
            for t in tubes
        ]
        
        return (
            self._remesh(parent_cut),
            [self._remesh(c) for c in children_cut],
            [self._remesh(t) for t in tubes_cut]
        )

    def _remesh(self, mesh: Trimesh, max_len=0.8):
        vertices, faces = subdivide_to_size(mesh.vertices, mesh.faces, max_len)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        return mesh
