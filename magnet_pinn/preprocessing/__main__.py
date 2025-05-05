import numpy as np

from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PointPreprocessing
)
from magnet_pinn.preprocessing.cli import (
    parse_arguments, print_report
)


args = parse_arguments()
print(args)

if args.preprocessing_type == "grid":
    prep = GridPreprocessing(
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
    )
elif args.preprocessing_type == "point":
    prep = PointPreprocessing(
        args.batches,
        args.antenna,
        args.output,
        field_dtype=args.field_dtype
    )
else:
    raise ValueError("Invalid preprocessing type")

print_report(args, prep)
prep.process_simulations(simulations=args.simulations)
