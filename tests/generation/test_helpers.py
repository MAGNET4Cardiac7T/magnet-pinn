from pathlib import Path
from argparse import Namespace

import pytest

from magnet_pinn.generator.cli.helpers import (
    print_report,
    validate_arguments,
    _print_tissue_report,
    _print_custom_report,
    _print_properties_report,
    _print_workflow_report,
)


def test_print_report_tissue_phantom_without_output_path(capsys):
    """
    Case when we print report for tissue phantom without output path
    """
    args = Namespace(
        phantom_type="tissue",
        seed=42,
        output=Path("/test/output"),
        num_children_blobs=10,
        initial_blob_radius=5.0,
        blob_radius_decrease=0.8,
        num_tubes=3,
        relative_tube_min_radius=0.1,
        relative_tube_max_radius=0.5,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        transforms_mode="default",
    )
    print_report(args)
    captured = capsys.readouterr()

    assert "Data Generation Report" in captured.out
    assert "Phantom Type: tissue" in captured.out
    assert "Seed: 42" in captured.out
    assert "Output Directory: /test/output" in captured.out
    assert "Number of children blobs: 10" in captured.out
    assert "Initial blob radius: 5.0" in captured.out
    assert "Transforms mode: default" in captured.out


def test_print_report_tissue_phantom_with_output_path(capsys):
    """
    Case when we print report for tissue phantom with output path
    """
    args = Namespace(
        phantom_type="tissue",
        seed=42,
        output=Path("/test/output"),
        num_children_blobs=10,
        initial_blob_radius=5.0,
        blob_radius_decrease=0.8,
        num_tubes=3,
        relative_tube_min_radius=0.1,
        relative_tube_max_radius=0.5,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        transforms_mode="default",
    )
    output_path = Path("/test/output/result.h5")
    print_report(args, output_path)
    captured = capsys.readouterr()

    assert "Saved to:" in captured.out
    assert "result.h5" in captured.out


def test_print_report_custom_phantom(capsys):
    """
    Case when we print report for custom phantom
    """
    args = Namespace(
        phantom_type="custom",
        seed=123,
        output=Path("/custom/output"),
        stl_mesh_path=Path("/path/to/mesh.stl"),
        num_children_blobs=5,
        blob_radius_decrease=0.7,
        num_tubes=2,
        relative_tube_min_radius=0.2,
        relative_tube_max_radius=0.6,
        sample_children_only_inside=True,
        child_blobs_batch_size=100,
        density_min=1.0,
        density_max=3.0,
        conductivity_min=0.5,
        conductivity_max=2.0,
        permittivity_min=60.0,
        permittivity_max=90.0,
        transforms_mode="advanced",
    )
    print_report(args)
    captured = capsys.readouterr()

    assert "Phantom Type: custom" in captured.out
    assert "STL mesh path: /path/to/mesh.stl" in captured.out
    assert "Sample children only inside: True" in captured.out
    assert "Child blobs batch size: 100" in captured.out


def test_print_report_unknown_phantom_type(capsys):
    """
    Case when we print report for unknown phantom type
    """
    args = Namespace(
        phantom_type="unknown",
        seed=1,
        output=Path("/output"),
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        transforms_mode="default",
    )
    print_report(args)
    captured = capsys.readouterr()

    assert "Phantom Type: unknown" in captured.out
    assert "Structure Configuration:" in captured.out


def test_print_tissue_report(capsys):
    """
    Case when we print tissue-specific configuration
    """
    args = Namespace(
        num_children_blobs=15,
        initial_blob_radius=7.5,
        blob_radius_decrease=0.9,
        num_tubes=4,
        relative_tube_min_radius=0.15,
        relative_tube_max_radius=0.45,
        x_min=-5,
        x_max=5,
        y_min=-8,
        y_max=8,
        z_min=-12,
        z_max=12,
    )
    _print_tissue_report(args)
    captured = capsys.readouterr()

    assert "Number of children blobs: 15" in captured.out
    assert "Initial blob radius: 7.5" in captured.out
    assert "Blob radius decrease factor: 0.9" in captured.out
    assert "Number of tubes: 4" in captured.out
    assert "Relative tube radius range: [0.15, 0.45]" in captured.out
    assert "X: [-5, 5]" in captured.out
    assert "Y: [-8, 8]" in captured.out
    assert "Z: [-12, 12]" in captured.out


def test_print_custom_report(capsys):
    """
    Case when we print custom phantom-specific configuration
    """
    args = Namespace(
        stl_mesh_path=Path("/custom/mesh.stl"),
        num_children_blobs=20,
        blob_radius_decrease=0.85,
        num_tubes=5,
        relative_tube_min_radius=0.25,
        relative_tube_max_radius=0.55,
        sample_children_only_inside=False,
        child_blobs_batch_size=200,
    )
    _print_custom_report(args)
    captured = capsys.readouterr()

    assert "STL mesh path: /custom/mesh.stl" in captured.out
    assert "Number of children blobs: 20" in captured.out
    assert "Blob radius decrease factor: 0.85" in captured.out
    assert "Number of tubes: 5" in captured.out
    assert "Relative tube radius range: [0.25, 0.55]" in captured.out
    assert "Sample children only inside: False" in captured.out
    assert "Child blobs batch size: 200" in captured.out


def test_print_properties_report(capsys):
    """
    Case when we print physical properties configuration
    """
    args = Namespace(
        density_min=0.8,
        density_max=2.5,
        conductivity_min=0.3,
        conductivity_max=1.8,
        permittivity_min=55.0,
        permittivity_max=95.0,
    )
    _print_properties_report(args)
    captured = capsys.readouterr()

    assert "Density range: [0.8, 2.5]" in captured.out
    assert "Conductivity range: [0.3, 1.8]" in captured.out
    assert "Permittivity range: [55.0, 95.0]" in captured.out


def test_print_workflow_report(capsys):
    """
    Case when we print workflow configuration
    """
    args = Namespace(transforms_mode="custom")
    _print_workflow_report(args)
    captured = capsys.readouterr()

    assert "Transforms mode: custom" in captured.out


def test_validate_arguments_valid_tissue_args():
    """
    Case when we validate valid tissue phantom arguments
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        num_children_blobs=10,
        blob_radius_decrease=0.8,
        num_tubes=3,
        relative_tube_min_radius=0.1,
        relative_tube_max_radius=0.5,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    validate_arguments(args)


def test_validate_arguments_valid_custom_args(generation_output_dir_path):
    """
    Case when we validate valid custom phantom arguments
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    args = Namespace(
        phantom_type="custom",
        density_min=1.0,
        density_max=3.0,
        conductivity_min=0.5,
        conductivity_max=2.0,
        permittivity_min=60.0,
        permittivity_max=90.0,
        num_children_blobs=5,
        blob_radius_decrease=0.7,
        num_tubes=2,
        relative_tube_min_radius=0.2,
        relative_tube_max_radius=0.6,
        stl_mesh_path=stl_file,
        child_blobs_batch_size=100,
    )
    validate_arguments(args)


def test_validate_arguments_density_min_greater_than_max():
    """
    Case when density_min is greater than or equal to density_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=2.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
    )
    with pytest.raises(ValueError, match="density_min must be less than density_max"):
        validate_arguments(args)


def test_validate_arguments_density_min_equal_to_max():
    """
    Case when density_min equals density_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=2.0,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
    )
    with pytest.raises(ValueError, match="density_min must be less than density_max"):
        validate_arguments(args)


def test_validate_arguments_conductivity_min_greater_than_max():
    """
    Case when conductivity_min is greater than or equal to conductivity_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=2.0,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
    )
    with pytest.raises(ValueError, match="conductivity_min must be less than conductivity_max"):
        validate_arguments(args)


def test_validate_arguments_conductivity_min_equal_to_max():
    """
    Case when conductivity_min equals conductivity_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=1.5,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
    )
    with pytest.raises(ValueError, match="conductivity_min must be less than conductivity_max"):
        validate_arguments(args)


def test_validate_arguments_permittivity_min_greater_than_max():
    """
    Case when permittivity_min is greater than or equal to permittivity_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=120.0,
        permittivity_max=100.0,
    )
    with pytest.raises(ValueError, match="permittivity_min must be less than permittivity_max"):
        validate_arguments(args)


def test_validate_arguments_permittivity_min_equal_to_max():
    """
    Case when permittivity_min equals permittivity_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=100.0,
        permittivity_max=100.0,
    )
    with pytest.raises(ValueError, match="permittivity_min must be less than permittivity_max"):
        validate_arguments(args)


def test_validate_arguments_negative_num_children_blobs():
    """
    Case when num_children_blobs is negative
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        num_children_blobs=-5,
    )
    with pytest.raises(ValueError, match="num_children_blobs must be non-negative"):
        validate_arguments(args)


def test_validate_arguments_zero_num_children_blobs():
    """
    Case when num_children_blobs is zero
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        num_children_blobs=0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    validate_arguments(args)


def test_validate_arguments_blob_radius_decrease_zero():
    """
    Case when blob_radius_decrease is zero
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        blob_radius_decrease=0.0,
    )
    with pytest.raises(ValueError, match="blob_radius_decrease must be in range"):
        validate_arguments(args)


def test_validate_arguments_blob_radius_decrease_negative():
    """
    Case when blob_radius_decrease is negative
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        blob_radius_decrease=-0.5,
    )
    with pytest.raises(ValueError, match="blob_radius_decrease must be in range"):
        validate_arguments(args)


def test_validate_arguments_blob_radius_decrease_one():
    """
    Case when blob_radius_decrease equals one
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        blob_radius_decrease=1.0,
    )
    with pytest.raises(ValueError, match="blob_radius_decrease must be in range"):
        validate_arguments(args)


def test_validate_arguments_blob_radius_decrease_greater_than_one():
    """
    Case when blob_radius_decrease is greater than one
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        blob_radius_decrease=1.5,
    )
    with pytest.raises(ValueError, match="blob_radius_decrease must be in range"):
        validate_arguments(args)


def test_validate_arguments_negative_num_tubes():
    """
    Case when num_tubes is negative
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        num_tubes=-3,
    )
    with pytest.raises(ValueError, match="num_tubes must be non-negative"):
        validate_arguments(args)


def test_validate_arguments_zero_num_tubes():
    """
    Case when num_tubes is zero
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        num_tubes=0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    validate_arguments(args)


def test_validate_arguments_relative_tube_min_radius_zero():
    """
    Case when relative_tube_min_radius is zero
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_min_radius=0.0,
    )
    with pytest.raises(ValueError, match="relative_tube_min_radius must be positive"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_min_radius_negative():
    """
    Case when relative_tube_min_radius is negative
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_min_radius=-0.1,
    )
    with pytest.raises(ValueError, match="relative_tube_min_radius must be positive"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_max_radius_zero():
    """
    Case when relative_tube_max_radius is zero
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_max_radius=0.0,
    )
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in range"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_max_radius_negative():
    """
    Case when relative_tube_max_radius is negative
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_max_radius=-0.5,
    )
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in range"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_max_radius_one():
    """
    Case when relative_tube_max_radius equals one
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_max_radius=1.0,
    )
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in range"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_max_radius_greater_than_one():
    """
    Case when relative_tube_max_radius is greater than one
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_max_radius=1.5,
    )
    with pytest.raises(ValueError, match="relative_tube_max_radius must be in range"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_min_greater_than_max():
    """
    Case when relative_tube_min_radius is greater than or equal to relative_tube_max_radius
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_min_radius=0.6,
        relative_tube_max_radius=0.5,
    )
    with pytest.raises(ValueError, match="relative_tube_min_radius must be less than relative_tube_max_radius"):
        validate_arguments(args)


def test_validate_arguments_relative_tube_min_equal_to_max():
    """
    Case when relative_tube_min_radius equals relative_tube_max_radius
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        relative_tube_min_radius=0.5,
        relative_tube_max_radius=0.5,
    )
    with pytest.raises(ValueError, match="relative_tube_min_radius must be less than relative_tube_max_radius"):
        validate_arguments(args)


def test_validate_tissue_arguments_negative_initial_blob_radius():
    """
    Case when initial_blob_radius is negative
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=-5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="initial_blob_radius must be positive"):
        validate_arguments(args)


def test_validate_tissue_arguments_zero_initial_blob_radius():
    """
    Case when initial_blob_radius is zero
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=0.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="initial_blob_radius must be positive"):
        validate_arguments(args)


def test_validate_tissue_arguments_x_min_greater_than_max():
    """
    Case when x_min is greater than or equal to x_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=15,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="x_min must be less than x_max"):
        validate_arguments(args)


def test_validate_tissue_arguments_x_min_equal_to_max():
    """
    Case when x_min equals x_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="x_min must be less than x_max"):
        validate_arguments(args)


def test_validate_tissue_arguments_y_min_greater_than_max():
    """
    Case when y_min is greater than or equal to y_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=15,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="y_min must be less than y_max"):
        validate_arguments(args)


def test_validate_tissue_arguments_y_min_equal_to_max():
    """
    Case when y_min equals y_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="y_min must be less than y_max"):
        validate_arguments(args)


def test_validate_tissue_arguments_z_min_greater_than_max():
    """
    Case when z_min is greater than or equal to z_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=15,
        z_max=10,
    )
    with pytest.raises(ValueError, match="z_min must be less than z_max"):
        validate_arguments(args)


def test_validate_tissue_arguments_z_min_equal_to_max():
    """
    Case when z_min equals z_max
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=10,
        z_max=10,
    )
    with pytest.raises(ValueError, match="z_min must be less than z_max"):
        validate_arguments(args)


def test_validate_custom_arguments_stl_file_not_exists(generation_output_dir_path):
    """
    Case when STL mesh file does not exist
    """
    stl_file = generation_output_dir_path / "nonexistent.stl"

    args = Namespace(
        phantom_type="custom",
        density_min=1.0,
        density_max=3.0,
        conductivity_min=0.5,
        conductivity_max=2.0,
        permittivity_min=60.0,
        permittivity_max=90.0,
        stl_mesh_path=stl_file,
        child_blobs_batch_size=100,
    )
    with pytest.raises(ValueError, match="STL mesh file not found"):
        validate_arguments(args)


def test_validate_custom_arguments_stl_file_wrong_extension(generation_output_dir_path):
    """
    Case when STL mesh file has wrong extension
    """
    wrong_file = generation_output_dir_path / "mesh.obj"
    wrong_file.touch()

    args = Namespace(
        phantom_type="custom",
        density_min=1.0,
        density_max=3.0,
        conductivity_min=0.5,
        conductivity_max=2.0,
        permittivity_min=60.0,
        permittivity_max=90.0,
        stl_mesh_path=wrong_file,
        child_blobs_batch_size=100,
    )
    with pytest.raises(ValueError, match="STL mesh file must have .stl extension"):
        validate_arguments(args)


def test_validate_custom_arguments_child_blobs_batch_size_zero(generation_output_dir_path):
    """
    Case when child_blobs_batch_size is zero
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    args = Namespace(
        phantom_type="custom",
        density_min=1.0,
        density_max=3.0,
        conductivity_min=0.5,
        conductivity_max=2.0,
        permittivity_min=60.0,
        permittivity_max=90.0,
        stl_mesh_path=stl_file,
        child_blobs_batch_size=0,
    )
    with pytest.raises(ValueError, match="child_blobs_batch_size must be at least 1"):
        validate_arguments(args)


def test_validate_custom_arguments_child_blobs_batch_size_negative(generation_output_dir_path):
    """
    Case when child_blobs_batch_size is negative
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    args = Namespace(
        phantom_type="custom",
        density_min=1.0,
        density_max=3.0,
        conductivity_min=0.5,
        conductivity_max=2.0,
        permittivity_min=60.0,
        permittivity_max=90.0,
        stl_mesh_path=stl_file,
        child_blobs_batch_size=-5,
    )
    with pytest.raises(ValueError, match="child_blobs_batch_size must be at least 1"):
        validate_arguments(args)


def test_validate_arguments_no_optional_attributes():
    """
    Case when validate arguments without optional attributes
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    validate_arguments(args)


def test_validate_arguments_only_some_optional_attributes():
    """
    Case when validate arguments with only some optional attributes
    """
    args = Namespace(
        phantom_type="tissue",
        density_min=0.5,
        density_max=2.0,
        conductivity_min=0.1,
        conductivity_max=1.5,
        permittivity_min=50.0,
        permittivity_max=100.0,
        num_children_blobs=10,
        initial_blob_radius=5.0,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        z_min=-10,
        z_max=10,
    )
    validate_arguments(args)
