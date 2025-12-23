
.. _gen_meshes:

-----------------------
Generate Meshes
-----------------------

In the following it is shown how to generate your own sample data.
The following code snippet generates stl files with the given Tissue data.

Therefore we need to define how the Tissue should look like, e.g. how many blobs and tubes should be generated.
Then we generate the 3D structures and save thm as meshes.

.. code-block:: python

    from numpy.random import default_rng
    from magnet_pinn.generator.io import MeshWriter
    from magnet_pinn.generator.phantoms import Tissue
    from magnet_pinn.generator.samplers import PropertySampler
    from magnet_pinn.generator.transforms import ToMesh, MeshesCutout, MeshesCleaning, Compose

    # Step 1/4: Generate Tissue with blobs and tubes inside
    tissue = Tissue(
        num_children_blobs=3,
        initial_blob_radius=100,
        initial_blob_center_extent={
            "x": [-5, 5],
            "y": [-5, 5],
            "z": [-50, 50],
        },
        blob_radius_decrease_per_level=0.3,
        num_tubes=10,
        relative_tube_max_radius=0.1,
        relative_tube_min_radius=0.01
    )
    raw_3d_structures = tissue.generate(seed=42)

    # Step 2/4: Define the workflow for processing structures
    workflow = Compose([
        ToMesh(),
        MeshesCutout(),
        MeshesCleaning()
    ])
    meshes = workflow(raw_3d_structures)

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
    writer = MeshWriter("./gen_data/raw/tissue_meshes")
    writer.write(meshes, prop)
