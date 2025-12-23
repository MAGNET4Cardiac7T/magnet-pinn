from pathlib import Path

import pytest
import numpy as np

from magnet_pinn.preprocessing.cli import parse_arguments
from magnet_pinn.preprocessing.cli.cli import (
    BATCHES,
    ANTENNA_DIR,
    OUTPUT_DIR,
    FIEld_DTYPE,
    VOXEL_SIZE,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    Z_MIN,
    Z_MAX,
)


def test_cli_check_grid_default_command(monkeypatch):
    """
    General check for the grid command
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.preprocessing_type == "grid"


def test_cli_check_no_command(monkeypatch):
    """
    Case when no command is given
    """
    monkeypatch.setattr("sys.argv", ["script.py"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_batches_argmunets_for_one_value(
    monkeypatch, raw_central_batch_dir_path
):
    """
    Case when we pass a grid command and one batch directory
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "grid", "--batches", str(raw_central_batch_dir_path)]
    )
    args = parse_arguments()
    assert args.batches == [raw_central_batch_dir_path]


def test_grid_cli_check_batches_argmunets_for_multiple_values(
    monkeypatch, raw_central_batch_dir_path, raw_shifted_batch_dir_path
):
    """
    Case when we pass a grid command and multiple batch directories
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "grid",
            "--batches",
            str(raw_shifted_batch_dir_path),
            str(raw_central_batch_dir_path),
        ],
    )
    args = parse_arguments()
    assert args.batches == [raw_shifted_batch_dir_path, raw_central_batch_dir_path]


def test_grid_cli_check_batches_arguments_default_value(monkeypatch):
    """
    Case when we pass a grid command and no batch directories
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.batches == BATCHES


def test_grid_cli_check_given_antenna_value(monkeypatch, raw_antenna_dir_path):
    """
    Case when we give an antenna argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "grid", "--antenna", str(raw_antenna_dir_path)]
    )
    args = parse_arguments()
    assert args.antenna == raw_antenna_dir_path


def test_grid_cli_check_default_antenna_value(monkeypatch):
    """
    Case when we check a default antenna value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.antenna == ANTENNA_DIR


def test_grid_cli_check_given_output_value(monkeypatch, processed_batch_dir_path):
    """
    Case when we give an output argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "grid", "--output", str(processed_batch_dir_path)]
    )
    args = parse_arguments()
    assert args.output == processed_batch_dir_path


def test_grid_cli_check_default_output_value(monkeypatch):
    """
    Case when we check a default output value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.output == OUTPUT_DIR


def test_grid_cli_check_given_field_dtype_valid_value(monkeypatch):
    """
    Case when we give a field_dtype argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--field_dtype", "float64"])
    args = parse_arguments()
    assert args.field_dtype == np.float64


def test_grid_cli_check_given_field_dtype_invalid_value(monkeypatch):
    """
    Case when we give an invalid field_dtype argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "grid", "--field_dtype", "some_value"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_field_dtype_value(monkeypatch):
    """
    Case when we check a default field_dtype value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.field_dtype == FIEld_DTYPE


def test_grid_cli_check_simulations_one_given_value(monkeypatch):
    """
    Case when we give a simulations argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "grid", "--simulations", "simulation_1"]
    )
    args = parse_arguments()
    assert args.simulations == [Path("simulation_1")]


def test_grid_cli_check_simulations_multiple_given_values(monkeypatch):
    """
    Case when we give multiple simulations arguments
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "grid",
            "--simulations",
            "simulation_1",
            "simulation_2",
            "simulation_3",
        ],
    )
    args = parse_arguments()
    assert args.simulations == [
        Path("simulation_1"),
        Path("simulation_2"),
        Path("simulation_3"),
    ]


def test_grid_cli_check_simulations_default_value(monkeypatch):
    """
    Case when we check a default simulations value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.simulations is None


def test_grid_cli_check_voxel_size_given_valid_value(monkeypatch):
    """
    Case when we give a voxel_size argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--voxel_size", "0.1"])
    args = parse_arguments()
    assert args.voxel_size == 0.1


def test_grid_cli_check_voxel_size_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid voxel_size argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--voxel_size", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_voxel_size_value(monkeypatch):
    """
    Case when we check a default voxel_size value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.voxel_size == VOXEL_SIZE


def test_grid_cli_check_x_min_given_valid_value(monkeypatch):
    """
    Case when we give a x_min argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--x_min", "-100"])
    args = parse_arguments()
    assert args.x_min == -100


def test_grid_cli_check_x_min_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid x_min argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--x_min", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_x_min_value(monkeypatch):
    """
    Case when we check a default x_min value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.x_min == X_MIN


def test_grid_cli_check_x_max_given_valid_value(monkeypatch):
    """
    Case when we give a x_max argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--x_max", "100"])
    args = parse_arguments()
    assert args.x_max == 100


def test_grid_cli_check_x_max_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid x_max argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--x_max", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_x_max_value(monkeypatch):
    """
    Case when we check a default x_max value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.x_max == X_MAX


def test_grid_cli_check_y_min_given_valid_value(monkeypatch):
    """
    Case when we give a y_min argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--y_min", "-100"])
    args = parse_arguments()
    assert args.y_min == -100


def test_grid_cli_check_y_min_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid y_min argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--y_min", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_y_min_value(monkeypatch):
    """
    Case when we check a default y_min value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.y_min == Y_MIN


def test_grid_cli_check_y_max_given_valid_value(monkeypatch):
    """
    Case when we give a y_max argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--y_max", "100"])
    args = parse_arguments()
    assert args.y_max == 100


def test_grid_cli_check_y_max_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid y_max argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--y_max", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_y_max_value(monkeypatch):
    """
    Case when we check a default y_max value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.y_max == Y_MAX


def test_grid_cli_check_z_min_given_valid_value(monkeypatch):
    """
    Case when we give a z_min argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--z_min", "-100"])
    args = parse_arguments()
    assert args.z_min == -100


def test_grid_cli_check_z_min_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid z_min argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--z_min", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_z_min_value(monkeypatch):
    """
    Case when we check a default z_min value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.z_min == Z_MIN


def test_grid_cli_check_z_max_given_valid_value(monkeypatch):
    """
    Case when we give a z_max argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--z_max", "100"])
    args = parse_arguments()
    assert args.z_max == 100


def test_grid_cli_check_z_max_given_invalid_value(monkeypatch):
    """
    Case when we give an invalid z_max argument
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid", "--z_max", "some_value"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_grid_cli_check_default_z_max_value(monkeypatch):
    """
    Case when we check a default z_max value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "grid"])
    args = parse_arguments()
    assert args.z_max == Z_MAX


def test_cli_check_pointcloud_default_command(monkeypatch):
    """
    General check for the pointcloud command
    """
    monkeypatch.setattr("sys.argv", ["script.py", "pointcloud"])
    args = parse_arguments()
    assert args.preprocessing_type == "pointcloud"


def test_pointcloud_cli_check_batches_argmunets_for_one_value(
    monkeypatch, raw_central_batch_dir_path
):
    """
    Case when we pass a pointcloud command and one batch directory
    """
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "pointcloud", "--batches", str(raw_central_batch_dir_path)],
    )
    args = parse_arguments()
    assert args.batches == [raw_central_batch_dir_path]


def test_pointcloud_cli_check_batches_argmunets_for_multiple_values(
    monkeypatch, raw_central_batch_dir_path, raw_shifted_batch_dir_path
):
    """
    Case when we pass a pointcloud command and multiple batch directories
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "pointcloud",
            "--batches",
            str(raw_shifted_batch_dir_path),
            str(raw_central_batch_dir_path),
        ],
    )
    args = parse_arguments()
    assert args.batches == [raw_shifted_batch_dir_path, raw_central_batch_dir_path]


def test_pointcloud_cli_check_batches_arguments_default_value(monkeypatch):
    """
    Case when we pass a pointcloud command and no batch directories
    """
    monkeypatch.setattr("sys.argv", ["script.py", "pointcloud"])
    args = parse_arguments()
    assert args.batches == BATCHES


def test_pointcloud_cli_check_given_antenna_value(monkeypatch, raw_antenna_dir_path):
    """
    Case when we give an antenna argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "pointcloud", "--antenna", str(raw_antenna_dir_path)]
    )
    args = parse_arguments()
    assert args.antenna == raw_antenna_dir_path


def test_pointcloud_cli_check_default_antenna_value(monkeypatch):
    """
    Case when we check a default antenna value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "pointcloud"])
    args = parse_arguments()
    assert args.antenna == ANTENNA_DIR


def test_pointcloud_cli_check_given_output_value(monkeypatch, processed_batch_dir_path):
    """
    Case when we give an output argument
    """
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "pointcloud", "--output", str(processed_batch_dir_path)],
    )
    args = parse_arguments()
    assert args.output == processed_batch_dir_path


def test_pointcloud_cli_check_default_output_value(monkeypatch):
    """
    Case when we check a default output value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "pointcloud"])
    args = parse_arguments()
    assert args.output == OUTPUT_DIR


def test_pointcloud_cli_check_given_field_dtype_valid_value(monkeypatch):
    """
    Case when we give a field_dtype argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "pointcloud", "--field_dtype", "float64"]
    )
    args = parse_arguments()
    assert args.field_dtype == np.float64


def test_pointcloud_cli_check_given_field_dtype_invalid_value(monkeypatch):
    """
    Case when we give an invalid field_dtype argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "pointcloud", "--field_dtype", "some_value"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_pointcloud_cli_check_default_field_dtype_value(monkeypatch):
    """
    Case when we check a default field_dtype value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "pointcloud"])
    args = parse_arguments()
    assert args.field_dtype == FIEld_DTYPE


def test_pointcloud_cli_check_simulations_one_given_value(monkeypatch):
    """
    Case when we give a simulations argument
    """
    monkeypatch.setattr(
        "sys.argv", ["script.py", "pointcloud", "--simulations", "simulation_1"]
    )
    args = parse_arguments()
    assert args.simulations == [Path("simulation_1")]


def test_pointcloud_cli_check_simulations_multiple_given_values(monkeypatch):
    """
    Case when we give multiple simulations arguments
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "pointcloud",
            "--simulations",
            "simulation_1",
            "simulation_2",
            "simulation_3",
        ],
    )
    args = parse_arguments()
    assert args.simulations == [
        Path("simulation_1"),
        Path("simulation_2"),
        Path("simulation_3"),
    ]


def test_pointcloud_cli_check_simulations_default_value(monkeypatch):
    """
    Case when we check a default simulations value
    """
    monkeypatch.setattr("sys.argv", ["script.py", "pointcloud"])
    args = parse_arguments()
    assert args.simulations is None
