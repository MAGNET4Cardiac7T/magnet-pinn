"""
NAME
    helpers.py
DESCRIPTION
    Helper functions for the generator CLI module.
"""
from pathlib import Path
from argparse import Namespace


def print_report(args: Namespace, output_path: Path = None):
    """
    Print a report of the generation configuration and results.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.
    output_path : Path, optional
        Path where the output was saved. If provided, includes in report.
    """
    print("\n" + "="*60)
    print("Data Generation Report")
    print("="*60)
    
    print(f"\nPhantom Type: {args.phantom_type}")
    print(f"Seed: {args.seed}")
    print(f"Output Directory: {args.output}")
    
    if output_path:
        print(f"Saved to: {output_path.resolve()}")
    
    print("\n" + "-"*60)
    print("Structure Configuration:")
    print("-"*60)
    
    if args.phantom_type == "tissue":
        _print_tissue_report(args)
    elif args.phantom_type == "custom":
        _print_custom_report(args)
    
    print("\n" + "-"*60)
    print("Physical Properties Configuration:")
    print("-"*60)
    _print_properties_report(args)
    
    print("\n" + "-"*60)
    print("Workflow Configuration:")
    print("-"*60)
    _print_workflow_report(args)
    
    print("\n" + "="*60 + "\n")


def _print_tissue_report(args: Namespace):
    """Print tissue-specific configuration."""
    print(f"  Number of children blobs: {args.num_children_blobs}")
    print(f"  Initial blob radius: {args.initial_blob_radius}")
    print(f"  Blob radius decrease factor: {args.blob_radius_decrease}")
    print(f"  Number of tubes: {args.num_tubes}")
    print(f"  Relative tube radius range: [{args.relative_tube_min_radius}, {args.relative_tube_max_radius}]")
    print(f"  Parent blob center extent:")
    print(f"    X: [{args.x_min}, {args.x_max}]")
    print(f"    Y: [{args.y_min}, {args.y_max}]")
    print(f"    Z: [{args.z_min}, {args.z_max}]")


def _print_custom_report(args: Namespace):
    """Print custom phantom-specific configuration."""
    print(f"  STL mesh path: {args.stl_mesh_path}")
    print(f"  Number of children blobs: {args.num_children_blobs}")
    print(f"  Blob radius decrease factor: {args.blob_radius_decrease}")
    print(f"  Number of tubes: {args.num_tubes}")
    print(f"  Relative tube radius range: [{args.relative_tube_min_radius}, {args.relative_tube_max_radius}]")
    print(f"  Sample children only inside: {args.sample_children_only_inside}")
    print(f"  Child blobs batch size: {args.child_blobs_batch_size}")


def _print_properties_report(args: Namespace):
    """Print physical properties configuration."""
    print(f"  Density range: [{args.density_min}, {args.density_max}]")
    print(f"  Conductivity range: [{args.conductivity_min}, {args.conductivity_max}]")
    print(f"  Permittivity range: [{args.permittivity_min}, {args.permittivity_max}]")


def _print_workflow_report(args: Namespace):
    """Print workflow configuration using preset mode."""
    print(f"  Transforms mode: {args.transforms_mode}")


def validate_arguments(args: Namespace) -> None:
    """
    Validate parsed arguments and raise errors if invalid configurations are detected.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Raises
    ------
    ValueError
        If any argument values are invalid or conflicting.
    """
    if args.density_min >= args.density_max:
        raise ValueError("density_min must be less than density_max")
    
    if args.conductivity_min >= args.conductivity_max:
        raise ValueError("conductivity_min must be less than conductivity_max")
    
    if args.permittivity_min >= args.permittivity_max:
        raise ValueError("permittivity_min must be less than permittivity_max")
    
    # Validate common phantom structure arguments
    if hasattr(args, 'num_children_blobs') and args.num_children_blobs < 0:
        raise ValueError("num_children_blobs must be non-negative")
    
    if hasattr(args, 'blob_radius_decrease') and (args.blob_radius_decrease <= 0 or args.blob_radius_decrease >= 1):
        raise ValueError("blob_radius_decrease must be in range (0, 1)")
    
    if hasattr(args, 'num_tubes') and args.num_tubes < 0:
        raise ValueError("num_tubes must be non-negative")
    
    if hasattr(args, 'relative_tube_min_radius') and args.relative_tube_min_radius <= 0:
        raise ValueError("relative_tube_min_radius must be positive")
    
    if hasattr(args, 'relative_tube_max_radius') and (args.relative_tube_max_radius <= 0 or args.relative_tube_max_radius >= 1):
        raise ValueError("relative_tube_max_radius must be in range (0, 1)")
    
    if hasattr(args, 'relative_tube_min_radius') and hasattr(args, 'relative_tube_max_radius') and args.relative_tube_min_radius >= args.relative_tube_max_radius:
        raise ValueError("relative_tube_min_radius must be less than relative_tube_max_radius")
    
    # Validate phantom-specific arguments
    if args.phantom_type == "tissue":
        _validate_tissue_arguments(args)
    elif args.phantom_type == "custom":
        _validate_custom_arguments(args)


def _validate_tissue_arguments(args: Namespace):
    """Validate tissue phantom-specific arguments."""
    if args.initial_blob_radius <= 0:
        raise ValueError("initial_blob_radius must be positive")
    
    if args.x_min >= args.x_max:
        raise ValueError("x_min must be less than x_max")
    
    if args.y_min >= args.y_max:
        raise ValueError("y_min must be less than y_max")
    
    if args.z_min >= args.z_max:
        raise ValueError("z_min must be less than z_max")


def _validate_custom_arguments(args: Namespace):
    """Validate custom phantom-specific arguments."""
    if not args.stl_mesh_path.exists():
        raise ValueError(f"STL mesh file not found: {args.stl_mesh_path}")
    
    if not args.stl_mesh_path.suffix.lower() == '.stl':
        raise ValueError(f"STL mesh file must have .stl extension: {args.stl_mesh_path}")
    
    if args.child_blobs_batch_size < 1:
        raise ValueError("child_blobs_batch_size must be at least 1")
