"""
NAME
    cli.py
DESCRIPTION
    This module implements CLI interface for the preprocessing module.
"""
import argparse
from pathlib import Path

from natsort import natsorted


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="magnet_pinn.preprocessing",
        description="Preprocess the data of the magnetostatic simulation"
    )

    parser.add_argument(
        "--batches",
        nargs="+",
        type=str,
        help="Paths of batch directories, be default takes all batches in the directory `./data/raw/batches` by the batch directory name `batch_*`",
        default=natsorted(Path("./data/raw/batches").glob("batch_*"))
    )

    parser.add_argument(
        "--antenna",
        type=str,
        help="Path of the antenna directory, by default takes the directory `./data/raw/antenna`",
        default=Path("./data/raw/antenna")
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path of the output directory, by default takes the directory `./data/processed`",
        default=Path("./data/processed")
    )

    parser.add_argument(
        "--field_dtype",
        type=str,
        default="float32",
        help="Data type of the field"
    )

    parser.add_argument(
        "--sim_names",
        nargs="+",
        type=str,
        default=None,
        help="Names of the simulations we would like to preprocess, leave empty to preprocess all simulations"
    )

    subparsers = parser.add_subparsers(dest="preprocessing_type", description="Type of preprocessing data")
    subparsers.required = True

    grid_parser = subparsers.add_parser("grid", help="Consider data in the 3D grid form")
    grid_parser.add_argument(
        "--voxel_size",
        type=float,
        default=1.0,
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

    points_parser = subparsers.add_parser("pointcloud", help="Consider data as a point cloud")
    return parser.parse_args()


def print_general_report(args):
    print("Preprocessing repot:")
    print("Batches paths: ", args.batches)
    print("Antenna path: ", args.antenna)
    print("Output path: ", args.output)

    expected_sim_names = args.sim_names if args.sim_names else "All"
    print("Simulations to preprocess: ", expected_sim_names)
    print("Field data type: ", args.field_dtype)
    print("Preprocessing type: ", args.preprocessing_type)


def print_grid_report(args): 
    print_general_report(args)
    print("x_min: ", args.x_min)
    print("x_max: ", args.x_max)
    print("y_min: ", args.y_min)
    print("y_max: ", args.y_max)
    print("z_min: ", args.z_min)
    print("z_max: ", args.z_max)
    print("voxel size: ", args.voxel_size)
