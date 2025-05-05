"""
NAME
    cli.py
DESCRIPTION
    This module implements CLI interface for the preprocessing module.
"""
import argparse
from pathlib import Path

import numpy as np
from argparse import Namespace
from natsort import natsorted


BATCHES = natsorted(Path("./data/raw/batches").glob("batch_*"))
ANTENNA_DIR = Path("./data/raw/antenna")
OUTPUT_DIR = Path("./data/processed")
FIEld_DTYPE = np.float32
VOXEL_SIZE = 4.0
X_MIN = -240
X_MAX = 240
Y_MIN = -220
Y_MAX = 220
Z_MIN = -250
Z_MAX = 250


def parse_arguments() -> Namespace:
    """
    So, here is the function to parse the CLI arguments. It creates a global parser, which predefines the arguments
    for `batches`, `antenna`, `output`, `field_dtype`, and `sim_names`. This parent parse is inherited by the all parsers:
    the main one and grid/pointcloud subparsers.

    Returns:
    --------
    args: argparse.Namespace
        The parsed arguments
    """
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "-b",
        "--batches",
        nargs="+",
        type=Path,
        help="Paths of batch directories, by default is all of the directories in the `./data/raw/batches/` directory from the current user`s directory",
        default=BATCHES
    )
    global_parser.add_argument(
        "-a",
        "--antenna",
        type=Path,
        help="Path of the antenna directory, be default is `./data/raw/antenna` from the current user`s directory",
        default=ANTENNA_DIR
    )
    global_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path of the output directory, by default is `./data/processed` from the current user`s directory",
        default=OUTPUT_DIR
    )
    global_parser.add_argument(
        "-t",
        "--field_dtype",
        type=np.dtype,
        default=FIEld_DTYPE,
        help="Data type of the field, must be a string representative of the np.dtype, by default is np.float32"
    )
    global_parser.add_argument(
        "-s",
        "--simulations",
        nargs="+",
        type=Path,
        default=None,
        help="Paths/names of simulations to preprocess, by default gets `None` value and preprocesses all the simulations in the batch directories"
    )

    
    main_parser = argparse.ArgumentParser(
        prog="magnet_pinn.preprocessing",
        description="Preprocess the simulation data",
        parents=[global_parser]
    )

    
    subparsers = main_parser.add_subparsers(
        dest="preprocessing_type",
        title="Subcommands",
        description="Type of preprocessing data",
        help="Sub-command to run (grid or point cloud)"
    )
    subparsers.required = True

    grid_parser = subparsers.add_parser("grid", parents=[global_parser], help="Process data in grid form")
    grid_parser.add_argument(
        "--voxel_size",
        type=float,
        default=4.0,
        help="Size of the voxel, be default is 4.0"
    )
    grid_parser.add_argument(
        "--x_min",
        type=float,
        default=-240,
        help="Minimum x-coordinate, be default is -240"
    )
    grid_parser.add_argument(
        "--x_max",
        type=float,
        default=240,
        help="Maximum x-coordinate, be default is 240"
    )
    grid_parser.add_argument(
        "--y_min",
        type=float,
        default=-220,
        help="Minimum y-coordinate, be default is -220"
    )
    grid_parser.add_argument(
        "--y_max",
        type=float,
        default=220,
        help="Maximum y-coordinate, be default is 220"
    )
    grid_parser.add_argument(
        "--z_min",
        type=float,
        default=-250,
        help="Minimum z-coordinate, by default is -250"
    )
    grid_parser.add_argument(
        "--z_max",
        type=float,
        default=250,
        help="Maximum z-coordinate, by default is 250"
    )

    pointcloud_parser = subparsers.add_parser("pointcloud", parents=[global_parser], help="Process data as a point cloud")

    return main_parser.parse_args()
