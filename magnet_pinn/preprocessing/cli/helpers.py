from argparse import Namespace

from magnet_pinn.preprocessing.preprocessing import Preprocessing, GridPreprocessing


def print_report(args: Namespace, prep: Preprocessing):
    print("Preprocessing report:")
    print("Batches: ", len(args.batches))
    print("Overall simulations: ", len(prep.all_sim_paths))
    print("Chosen simulations: ", len(args.simulations) if args.simulations else "All")
    print("Antenna: ", args.antenna)
    print("Output: ", args.output)
    print("Field data type: ", args.field_dtype)
    print("Preprocessing type: ", args.preprocessing_type)

    if isinstance(prep, GridPreprocessing):
        _print_grid_report(args)


def _print_grid_report(args: Namespace):
    print("x_min: ", args.x_min)
    print("x_max: ", args.x_max)
    print("y_min: ", args.y_min)
    print("y_max: ", args.y_max)
    print("z_min: ", args.z_min)
    print("z_max: ", args.z_max)
    print("voxel size: ", args.voxel_size)
