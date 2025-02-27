"""
NAME
    cli.py
DESCRIPTION
    This module implements CLI interface for the preprocessing module.
"""
import argparse
from argparse import Namespace
from pathlib import Path

from natsort import natsorted

from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, Preprocessing
)


def parse_arguments():
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
        help="Paths of batch directories",
        default=natsorted(Path("./data/raw/batches").glob("batch_*"))
    )
    global_parser.add_argument(
        "-a",
        "--antenna",
        type=Path,
        help="Path of the antenna directory",
        default=Path("./data/raw/antenna")
    )
    global_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path of the output directory",
        default=Path("./data/processed")
    )
    global_parser.add_argument(
        "-t",
        "--field_dtype",
        type=str,
        default="float32",
        help="Data type of the field"
    )
    global_parser.add_argument(
        "-s",
        "--simulations",
        nargs="+",
        type=Path,
        default=None,
        help="Paths/names of simulations to preprocess"
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
        help="Sub-command to run (grid or pointcloud)"
    )
    subparsers.required = True

    grid_parser = subparsers.add_parser("grid", parents=[global_parser], help="Process data in grid form")
    grid_parser.add_argument(
        "--voxel_size",
        type=float,
        default=4.0,
        help="Size of the voxel"
    )
    grid_parser.add_argument(
        "--x_min",
        type=float,
        default=-240,
        help="Minimum x-coordinate"
    )
    grid_parser.add_argument(
        "--x_max",
        type=float,
        default=240,
        help="Maximum x-coordinate"
    )
    grid_parser.add_argument(
        "--y_min",
        type=float,
        default=-220,
        help="Minimum y-coordinate"
    )
    grid_parser.add_argument(
        "--y_max",
        type=float,
        default=220,
        help="Maximum y-coordinate"
    )
    grid_parser.add_argument(
        "--z_min",
        type=float,
        default=-250,
        help="Minimum z-coordinate"
    )
    grid_parser.add_argument(
        "--z_max",
        type=float,
        default=250,
        help="Maximum z-coordinate"
    )

    pointcloud_parser = subparsers.add_parser("pointcloud", parents=[global_parser], help="Process data as a point cloud")

    return main_parser.parse_args()
