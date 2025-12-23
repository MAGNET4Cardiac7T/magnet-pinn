"""
NAME
    cli.py
DESCRIPTION
    This module implements CLI interface for the generator module.
"""
import argparse
from pathlib import Path
from argparse import Namespace


# Default values
OUTPUT_DIR = Path("./gen_data")
SEED = 42
NUM_CHILDREN_BLOBS = 3
NUM_TUBES = 10
INITIAL_BLOB_RADIUS = 100.0
BLOB_RADIUS_DECREASE = 0.3
RELATIVE_TUBE_MIN_RADIUS = 0.01
RELATIVE_TUBE_MAX_RADIUS = 0.1
X_MIN = -5.0
X_MAX = 5.0
Y_MIN = -5.0
Y_MAX = 5.0
Z_MIN = -50.0
Z_MAX = 50.0

# Property sampling defaults
DENSITY_MIN = 400.0
DENSITY_MAX = 2000.0
CONDUCTIVITY_MIN = 0.0
CONDUCTIVITY_MAX = 2.5
PERMITTIVITY_MIN = 1.0
PERMITTIVITY_MAX = 71.0

# Custom mesh defaults
SAMPLE_CHILDREN_ONLY_INSIDE = False
CHILD_BLOBS_BATCH_SIZE = 1000000


def parse_arguments() -> Namespace:
    """
    Parse CLI arguments for the data generation module. Creates a global parser which predefines
    arguments shared across all phantom types (output, seed, properties), and subparsers for
    specific phantom types (tissue, custom).

    Returns:
    --------
    args: argparse.Namespace
        The parsed arguments
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    
    # Common arguments for all phantom types
    global_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Path of the output directory, by default is {OUTPUT_DIR} from the current user's directory",
        default=OUTPUT_DIR
    )
    global_parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible generation, by default is 42"
    )
    
    # Property sampling arguments
    global_parser.add_argument(
        "--density-min",
        type=float,
        default=DENSITY_MIN,
        help="Minimum density value for property sampling, by default is 400.0"
    )
    global_parser.add_argument(
        "--density-max",
        type=float,
        default=DENSITY_MAX,
        help="Maximum density value for property sampling, by default is 2000.0"
    )
    global_parser.add_argument(
        "--conductivity-min",
        type=float,
        default=CONDUCTIVITY_MIN,
        help="Minimum conductivity value for property sampling, by default is 0.0"
    )
    global_parser.add_argument(
        "--conductivity-max",
        type=float,
        default=CONDUCTIVITY_MAX,
        help="Maximum conductivity value for property sampling, by default is 2.5"
    )
    global_parser.add_argument(
        "--permittivity-min",
        type=float,
        default=PERMITTIVITY_MIN,
        help="Minimum permittivity value for property sampling, by default is 1.0"
    )
    global_parser.add_argument(
        "--permittivity-max",
        type=float,
        default=PERMITTIVITY_MAX,
        help="Maximum permittivity value for property sampling, by default is 71.0"
    )
    
    # Workflow configuration preset
    global_parser.add_argument(
        "--transforms-mode",
        choices=['none', 'all', 'no-clipping'],
        default='all',
        help=("Preset selection of transforms: 'none' for no transforms, 'all' for all transforms, "
              "'no-clipping' to skip tube and children clipping, default value is 'all'. For more fine-grained workflows please modify the code directly.")
    )
    
    # Structure generation arguments (common to tissue and custom)
    global_parser.add_argument(
        "--num-children-blobs",
        type=int,
        default=NUM_CHILDREN_BLOBS,
        help="Number of child blob structures to generate, by default is 3"
    )
    global_parser.add_argument(
        "--blob-radius-decrease",
        type=float,
        default=BLOB_RADIUS_DECREASE,
        help="Scaling factor for child blob radii relative to parent radius, by default is 0.3"
    )
    global_parser.add_argument(
        "--num-tubes",
        type=int,
        default=NUM_TUBES,
        help="Number of tube structures to generate, by default is 10"
    )
    global_parser.add_argument(
        "--relative-tube-min-radius",
        type=float,
        default=RELATIVE_TUBE_MIN_RADIUS,
        help="Minimum tube radius as fraction of parent blob radius, by default is 0.01"
    )
    global_parser.add_argument(
        "--relative-tube-max-radius",
        type=float,
        default=RELATIVE_TUBE_MAX_RADIUS,
        help="Maximum tube radius as fraction of parent blob radius, by default is 0.1"
    )
    
    main_parser = argparse.ArgumentParser(
        prog="magnet_pinn.generator",
        description="Generate phantom data for MRI simulations",
        parents=[global_parser]
    )
    
    subparsers = main_parser.add_subparsers(
        dest="phantom_type",
        title="Phantom Types",
        description="Type of phantom to generate",
        help="Sub-command to run (tissue or custom)"
    )
    subparsers.required = True
    
    # Tissue phantom subcommand
    tissue_parser = subparsers.add_parser(
        "tissue",
        parents=[global_parser],
        help="Generate tissue phantom with hierarchical blob structures and tubes"
    )
    tissue_parser.add_argument(
        "--initial-blob-radius",
        type=float,
        default=INITIAL_BLOB_RADIUS,
        help="Radius of the parent blob structure, by default is 100.0"
    )
    tissue_parser.add_argument(
        "--x-min",
        type=float,
        default=X_MIN,
        help="Minimum x-coordinate for parent blob center, by default is -5.0"
    )
    tissue_parser.add_argument(
        "--x-max",
        type=float,
        default=X_MAX,
        help="Maximum x-coordinate for parent blob center, by default is 5.0"
    )
    tissue_parser.add_argument(
        "--y-min",
        type=float,
        default=Y_MIN,
        help="Minimum y-coordinate for parent blob center, by default is -5.0"
    )
    tissue_parser.add_argument(
        "--y-max",
        type=float,
        default=Y_MAX,
        help="Maximum y-coordinate for parent blob center, by default is 5.0"
    )
    tissue_parser.add_argument(
        "--z-min",
        type=float,
        default=Z_MIN,
        help="Minimum z-coordinate for parent blob center, by default is -50.0"
    )
    tissue_parser.add_argument(
        "--z-max",
        type=float,
        default=Z_MAX,
        help="Maximum z-coordinate for parent blob center, by default is 50.0"
    )
    
    # Custom phantom subcommand
    custom_parser = subparsers.add_parser(
        "custom",
        parents=[global_parser],
        help="Generate custom phantom based on STL mesh with blob and tube structures"
    )
    custom_parser.add_argument(
        "--stl-mesh-path",
        type=Path,
        required=True,
        help="Path to the STL mesh file for the parent structure (required)"
    )
    custom_parser.add_argument(
        "--sample-children-only-inside",
        action="store_true",
        default=SAMPLE_CHILDREN_ONLY_INSIDE,
        help="Ensure child blobs are fully contained within parent mesh volume"
    )
    custom_parser.add_argument(
        "--child-blobs-batch-size",
        type=int,
        default=CHILD_BLOBS_BATCH_SIZE,
        help="Number of points to sample in batch for child blob placement, by default is 1000000"
    )
    
    return main_parser.parse_args()
