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
from magnet_pinn.generator.transforms import (
    Compose,
    ToMesh,
    MeshesTubesClipping,
    MeshesChildrenCutout,
    MeshesParentCutoutWithChildren,
    MeshesParentCutoutWithTubes,
    MeshesChildrenClipping,
)

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate meshes with a custom parent mesh and tubes/blobs inside."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gen_data/raw/custom_mesh",
        help="Directory to save the generated meshes and properties",
    )
    parser.add_argument(
        "--stl_mesh_path",
        type=str,
        default="./phantom.stl",
        help="Path to the STL file of the custom parent mesh",
    )
    parser.add_argument(
        "--num_children_blobs",
        type=int,
        default=3,
        help="Number of child blobs to generate",
    )
    parser.add_argument(
        "--num_tubes", type=int, default=10, help="Number of tubes to generate"
    )
    return parser.parse_args()


args = parse_args()


def generate_phantom(num_children_blobs, num_tubes, output_dir, seed):
    # Step 1/4: Generate stage object wit custom mesh as parent object and with blobs and tubes inside
    phantom = CustomPhantom(
        stl_mesh_path=args.stl_mesh_path,
        num_children_blobs=num_children_blobs,
        blob_radius_decrease_per_level=0.2,
        num_tubes=num_tubes,
        relative_tube_max_radius=0.08,
        relative_tube_min_radius=0.02,
    )
    raw_3d_structures = phantom.generate(seed=seed)

    print(f"Generated phantom with:")
    print(f"- Parent blob at {raw_3d_structures.parent.position}")
    print(f"- {len(raw_3d_structures.children)} children blobs")
    print(f"- {len(raw_3d_structures.tubes)} tubes")

    # Step 2/5: Convert structures into meshes
    phantom_meshes = ToMesh()(raw_3d_structures)

    # Step 3/5: Define the workflow for processing meshes
    workflow = Compose(
        [
            # MeshesTubesClipping(),
            # MeshesChildrenCutout(),
            # MeshesParentCutoutWithChildren(),
            # MeshesParentCutoutWithTubes(),
            # MeshesChildrenClipping()
        ]
    )

    processed_meshes = workflow(phantom_meshes)
    print(f"Processed phantom with:")
    print(f"- Parent mesh with {len(processed_meshes.parent.vertices)} vertices")
    print(f"- {len(processed_meshes.children)} children meshes")
    print(f"- {len(processed_meshes.tubes)} tube meshes")

    # Optinonally, if the user has a display available, show the generated blobs and tubes
    if os.environ.get("DISPLAY") and False:
        print(
            "We will show generated tubes and children meshes, restricted by the parent mesh"
        )
        trimesh.boolean.union(
            processed_meshes.tubes + processed_meshes.children,
            engine="manifold",
        ).show()

    # Step 4/5: Sample physical properties for the generated meshes
    prop_sampler = PropertySampler(
        {
            "density": {"min": 400, "max": 2000},
            "conductivity": {"min": 0, "max": 2.5},
            "permittivity": {"min": 1, "max": 71},
        }
    )
    prop = prop_sampler.sample_like(processed_meshes, rng=default_rng())

    # Step 5/5: Save the generated meshes and properties to files
    writer = MeshWriter(output_dir)
    writer.write(processed_meshes, prop)
    print(f"Meshes and properties saved to {Path(writer.dir).resolve()}")


seed_sampler = default_rng(args.seed)
seeds = seed_sampler.integers(
    0, 100000, size=(args.num_children_blobs + 1, args.num_tubes + 1)
)

for num_children_blobs in range(args.num_children_blobs + 1):
    for num_tubes in range(args.num_tubes + 1):
        seed = seeds[num_children_blobs, num_tubes]
        output_dir = f"{args.output_dir}/children_{num_children_blobs}_tubes_{num_tubes}_seed_{seed}"
        generate_phantom(num_children_blobs, num_tubes, output_dir, seed)
        print(
            f"Generated meshes with {num_children_blobs} children blobs and {num_tubes} tubes."
        )
