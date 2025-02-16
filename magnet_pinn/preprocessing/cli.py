import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="magnet_pinn.preprocessing",
        description="Preprocess the data of the magnetostatic simulation"
    )

    parser.add_argument(
        "--batches_dir_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths of batch directories"
    )

    parser.add_argument(
        "--antenna_dir_path",
        type=str,
        required=True,
        help="Path of the antenna directory"
    )

    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Path of the output directory"
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

    subparsers = parser.add_subparsers(dest="command", description="Type of preprocessing")

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

