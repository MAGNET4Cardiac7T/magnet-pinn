"""
NAME
    meshes_generation.py
DESCRIPTION
    Example of how to generate experiments data by using the `src.generation` module and by having a custom mesh as a parent structure.
"""
import os
from pathlib import Path

import trimesh
from numpy.random import default_rng

from magnet_pinn.generator.io import MeshWriter
from magnet_pinn.generator.phantoms import CustomPhantom
from magnet_pinn.generator.samplers import PropertySampler
from magnet_pinn.generator.transforms import ToMesh, MeshesClipping, MeshesCleaning, Compose


# Step 1/4: Generate stage object wit custom mesh as parent object and with blobs and tubes inside
phantom = CustomPhantom(
    stl_mesh_path="./phantom.stl",
    num_children_blobs=5,
    blob_radius_decrease_per_level=0.2,
    num_tubes=10,
    relative_tube_max_radius=0.08,
    relative_tube_min_radius=0.02
)
raw_3d_structures = phantom.generate(seed=42)

print(f"Generated phantom with:")
print(f"- Parent blob at {raw_3d_structures.parent.position}")
print(f"- {len(raw_3d_structures.children)} children blobs")
print(f"- {len(raw_3d_structures.tubes)} tubes")

# Step 2/4: Define the workflow for processing structures
workflow = Compose([
    ToMesh(),
    MeshesClipping(),
    MeshesCleaning()
])
meshes = workflow(raw_3d_structures)
print(f"Processed phantom with:")
print(f"- Parent mesh with {len(meshes.parent.vertices)} vertices")
print(f"- {len(meshes.children)} children meshes")
print(f"- {len(meshes.tubes)} tube meshes")

# Optinonally, if the user has a display available, show the generated blobs and tubes
if os.environ.get("DISPLAY"):
    print("We will show generated tubes and children meshes, restricted by the parent mesh")
    trimesh.boolean.union(
        meshes.tubes + meshes.children,
        engine='manifold',
    ).show()

# Step 3/4: Sample physical properties for the generated meshes
prop_sampler = PropertySampler(
    {
        "density": {
            "min": 400,
            "max": 2000
        },
        "conductivity": {
            "min": 0,
            "max": 2.5
        },
        "permittivity": {
            "min": 1,
            "max": 71
        }
    }
)
prop = prop_sampler.sample_like(meshes, rng=default_rng())

# Step 4/4: Save the generated meshes and properties to files
writer = MeshWriter("./gen_data/raw/custom_mesh")
writer.write(meshes, prop)
print(f"Meshes and properties saved to {Path(writer.dir).resolve()}")
