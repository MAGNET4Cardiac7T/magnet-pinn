from magnet_pinn.preprocessing.cli import (
    parse_arguments, print_grid_report, print_general_report
)
from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PointPreprocessing
)


args = parse_arguments()

if args.preprocessing_type == "grid":
    print_grid_report(args)
    GridPreprocessing(
        args.batches,
        args.antenna,
        args.output,
        field_dtype=args.field_dtype,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        voxel_size=args.voxel_size
    ).process_simulations(args.sim_names)
elif args.preprocessing_type == "point":
    print_general_report(args)
    PointPreprocessing(
        args.batches,
        args.antenna,
        args.output,
        field_dtype=args.field_dtype,
        sim_names=args.sim_names
    ).process_simulations(args.sim_names)
else:
    pass
