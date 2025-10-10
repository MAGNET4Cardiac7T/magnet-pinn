"""
NAME
    __main__.py
DESCRIPTION
    Entry point for the magnet_pinn.generator module.
    This module orchestrates the phantom generation pipeline using CLI arguments.
"""
import logging
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from .io import MeshWriter
from .transforms import (
    Compose,
    ToMesh,
    MeshesTubesClipping,
    MeshesChildrenCutout,
    MeshesParentCutoutWithChildren,
    MeshesParentCutoutWithTubes,
    MeshesChildrenClipping,
    MeshesCleaning
)
from .cli.cli import parse_arguments
from .samplers import PropertySampler
from .phantoms import Tissue, CustomPhantom
from .cli.helpers import print_report, validate_arguments


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_property_sampler(args):
    """
    Create a PropertySampler from CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    PropertySampler
        Configured property sampler.
    """
    return PropertySampler({
        "density": {
            "min": args.density_min,
            "max": args.density_max
        },
        "conductivity": {
            "min": args.conductivity_min,
            "max": args.conductivity_max
        },
        "permittivity": {
            "min": args.permittivity_min,
            "max": args.permittivity_max
        }
    })


def create_tissue_phantom(args):
    """
    Create a Tissue phantom from CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    Tissue
        Configured tissue phantom generator.
    """
    initial_blob_center_extent = np.array([
        [args.x_min, args.x_max],
        [args.y_min, args.y_max],
        [args.z_min, args.z_max],
    ])
    
    return Tissue(
        num_children_blobs=args.num_children_blobs,
        initial_blob_radius=args.initial_blob_radius,
        initial_blob_center_extent=initial_blob_center_extent,
        blob_radius_decrease_per_level=args.blob_radius_decrease,
        num_tubes=args.num_tubes,
        relative_tube_max_radius=args.relative_tube_max_radius,
        relative_tube_min_radius=args.relative_tube_min_radius
    )


def create_custom_phantom(args):
    """
    Create a CustomPhantom from CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    CustomPhantom
        Configured custom phantom generator.
    """
    return CustomPhantom(
        stl_mesh_path=str(args.stl_mesh_path),
        num_children_blobs=args.num_children_blobs,
        blob_radius_decrease_per_level=args.blob_radius_decrease,
        num_tubes=args.num_tubes,
        relative_tube_max_radius=args.relative_tube_max_radius,
        relative_tube_min_radius=args.relative_tube_min_radius,
        sample_children_only_inside=args.sample_children_only_inside
    )


def create_workflow(args):
    """
    Create the mesh processing workflow from CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    Compose
        Configured transformation workflow.
    """
    mode = args.transforms_mode

    if mode == 'none':
        return Compose([])
    
    transforms = [
        MeshesChildrenCutout(),
        MeshesParentCutoutWithChildren(),
        MeshesParentCutoutWithTubes(),
    ]

    if mode == "all":
        transforms.extend([
            MeshesTubesClipping(),
            MeshesChildrenClipping()
        ])

    return Compose(transforms)


def generate_single_phantom(phantom_generator, property_sampler, workflow, args, output_dir: Path, seed: int):
    """
    Generate a single phantom with all processing steps.

    Parameters
    ----------
    phantom_generator : Phantom
        The phantom generator (Tissue or CustomPhantom).
    property_sampler : PropertySampler
        The property sampler for material properties.
    workflow : Compose
        The mesh processing workflow.
    args : Namespace
        Parsed command-line arguments.
    output_dir : Path
        Directory to save the output.
    seed : int
        Random seed for this phantom generation.

    Returns
    -------
    Path
        Path where the phantom was saved.
    """
    logger.info(f"Generating phantom with seed {seed}")
    
    # Step 1: Generate raw structures
    logger.info("Step 1/5: Generating raw 3D structures")
    if args.phantom_type == "custom":
        raw_structures = phantom_generator.generate(
            seed=seed,
            child_blobs_batch_size=args.child_blobs_batch_size
        )
    else:
        raw_structures = phantom_generator.generate(seed=seed)
    
    logger.info(f"Generated phantom with {len(raw_structures.children)} children and {len(raw_structures.tubes)} tubes")
    
    # Step 2: Convert to meshes
    logger.info("Step 2/5: Converting structures to meshes")
    phantom_meshes = ToMesh()(raw_structures)
    
    # Step 3: Process meshes
    logger.info("Step 3/5: Processing meshes through workflow")
    processed_meshes = workflow(phantom_meshes)
    logger.info(f"Processed phantom: parent has {len(processed_meshes.parent.vertices)} vertices")
    
    # Step 4: Sample properties
    logger.info("Step 4/5: Sampling physical properties")
    rng = default_rng(seed)
    properties = property_sampler.sample_like(processed_meshes, rng=rng)
    
    # Step 5: Write output
    logger.info("Step 5/5: Writing output files")
    writer = MeshWriter(output_dir)
    writer.write(processed_meshes, properties)
    
    logger.info(f"Successfully saved phantom to {output_dir.resolve()}")
    return output_dir


def main():
    """
    Main entry point for the generator CLI.
    
    Parses arguments, validates configuration, creates phantom generator,
    and orchestrates the generation pipeline.
    """
    # Parse and validate arguments
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        return 1
    
    # Create phantom generator
    if args.phantom_type == "tissue":
        phantom_generator = create_tissue_phantom(args)
    elif args.phantom_type == "custom":
        phantom_generator = create_custom_phantom(args)
    else:
        logger.error(f"Unknown phantom type: {args.phantom_type}")
        return 1
    
    # Create property sampler and workflow
    property_sampler = create_property_sampler(args)
    workflow = create_workflow(args)
    
    # Generate phantom
    logger.info("Starting phantom generation")
    
    try:
        # Generate the phantom
        output_path = generate_single_phantom(
            phantom_generator=phantom_generator,
            property_sampler=property_sampler,
            workflow=workflow,
            args=args,
            output_dir=args.output,
            seed=args.seed
        )
        
        # Print report
        print_report(args, output_path)
        
        logger.info("Successfully generated phantom")
        
    except Exception as e:
        logger.error(f"Error during phantom generation: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
