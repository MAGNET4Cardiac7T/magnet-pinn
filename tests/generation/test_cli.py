import pytest

from magnet_pinn.generator.cli.cli import parse_arguments
from magnet_pinn.generator.cli.cli import (
    OUTPUT_DIR, SEED, NUM_CHILDREN_BLOBS, NUM_TUBES,
    BLOB_RADIUS_DECREASE,
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
    DENSITY_MIN, DENSITY_MAX, CONDUCTIVITY_MIN, CONDUCTIVITY_MAX,
    PERMITTIVITY_MIN, PERMITTIVITY_MAX, SAMPLE_CHILDREN_ONLY_INSIDE,
    CHILD_BLOBS_BATCH_SIZE
)


def test_cli_check_no_command(monkeypatch):
    """
    Case when no command is given
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_custom_cli_check_default_output_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default output value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.output == OUTPUT_DIR


def test_tissue_cli_check_given_seed_value(monkeypatch):
    """
    Case when we give a seed argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--seed", "123"
        ]
    )
    args = parse_arguments()
    assert args.seed == 123


def test_tissue_cli_check_seed_invalid_value(monkeypatch):
    """
    Case when we give an invalid seed argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--seed", "invalid"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_tissue_cli_check_default_seed_value(monkeypatch):
    """
    Case when we check a default seed value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.seed == SEED


def test_custom_cli_check_given_seed_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a seed argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--seed", "456"
        ]
    )
    args = parse_arguments()
    assert args.seed == 456


def test_custom_cli_check_default_seed_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default seed value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.seed == SEED


def test_tissue_cli_check_given_density_min_value(monkeypatch):
    """
    Case when we give a density_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--density-min", "500.0"
        ]
    )
    args = parse_arguments()
    assert args.density_min == 500.0


def test_tissue_cli_check_density_min_invalid_value(monkeypatch):
    """
    Case when we give an invalid density_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--density-min", "invalid"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_tissue_cli_check_default_density_min_value(monkeypatch):
    """
    Case when we check a default density_min value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.density_min == DENSITY_MIN


def test_tissue_cli_check_given_density_max_value(monkeypatch):
    """
    Case when we give a density_max argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--density-max", "2500.0"
        ]
    )
    args = parse_arguments()
    assert args.density_max == 2500.0


def test_tissue_cli_check_default_density_max_value(monkeypatch):
    """
    Case when we check a default density_max value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.density_max == DENSITY_MAX


def test_tissue_cli_check_given_conductivity_min_value(monkeypatch):
    """
    Case when we give a conductivity_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--conductivity-min", "0.5"
        ]
    )
    args = parse_arguments()
    assert args.conductivity_min == 0.5


def test_tissue_cli_check_default_conductivity_min_value(monkeypatch):
    """
    Case when we check a default conductivity_min value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.conductivity_min == CONDUCTIVITY_MIN


def test_tissue_cli_check_given_conductivity_max_value(monkeypatch):
    """
    Case when we give a conductivity_max argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--conductivity-max", "3.0"
        ]
    )
    args = parse_arguments()
    assert args.conductivity_max == 3.0


def test_tissue_cli_check_default_conductivity_max_value(monkeypatch):
    """
    Case when we check a default conductivity_max value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.conductivity_max == CONDUCTIVITY_MAX


def test_tissue_cli_check_given_permittivity_min_value(monkeypatch):
    """
    Case when we give a permittivity_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--permittivity-min", "5.0"
        ]
    )
    args = parse_arguments()
    assert args.permittivity_min == 5.0


def test_tissue_cli_check_default_permittivity_min_value(monkeypatch):
    """
    Case when we check a default permittivity_min value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.permittivity_min == PERMITTIVITY_MIN


def test_tissue_cli_check_given_permittivity_max_value(monkeypatch):
    """
    Case when we give a permittivity_max argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--permittivity-max", "80.0"
        ]
    )
    args = parse_arguments()
    assert args.permittivity_max == 80.0


def test_tissue_cli_check_default_permittivity_max_value(monkeypatch):
    """
    Case when we check a default permittivity_max value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.permittivity_max == PERMITTIVITY_MAX


def test_tissue_cli_check_given_num_children_blobs_value(monkeypatch):
    """
    Case when we give a num_children_blobs argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--num-children-blobs", "5"
        ]
    )
    args = parse_arguments()
    assert args.num_children_blobs == 5


def test_tissue_cli_check_num_children_blobs_invalid_value(monkeypatch):
    """
    Case when we give an invalid num_children_blobs argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--num-children-blobs", "invalid"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_tissue_cli_check_default_num_children_blobs_value(monkeypatch):
    """
    Case when we check a default num_children_blobs value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.num_children_blobs == NUM_CHILDREN_BLOBS


def test_tissue_cli_check_given_blob_radius_decrease_value(monkeypatch):
    """
    Case when we give a blob_radius_decrease argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--blob-radius-decrease", "0.5"
        ]
    )
    args = parse_arguments()
    assert args.blob_radius_decrease == 0.5


def test_tissue_cli_check_default_blob_radius_decrease_value(monkeypatch):
    """
    Case when we check a default blob_radius_decrease value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.blob_radius_decrease == BLOB_RADIUS_DECREASE


def test_tissue_cli_check_given_num_tubes_value(monkeypatch):
    """
    Case when we give a num_tubes argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--num-tubes", "15"
        ]
    )
    args = parse_arguments()
    assert args.num_tubes == 15


def test_tissue_cli_check_default_num_tubes_value(monkeypatch):
    """
    Case when we check a default num_tubes value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.num_tubes == NUM_TUBES


def test_custom_cli_check_transforms_mode_default(monkeypatch, generation_output_dir_path):
    """
    Check default transforms_mode is 'all' for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "custom", "--stl-mesh-path", str(stl_file)]
    )
    args = parse_arguments()
    assert args.transforms_mode == 'all'


def test_custom_cli_check_transforms_mode_none(monkeypatch, generation_output_dir_path):
    """
    Check transforms_mode 'none' for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "custom", "--stl-mesh-path", str(stl_file), "--transforms-mode", "none"]
    )
    args = parse_arguments()
    assert args.transforms_mode == 'none'


def test_custom_cli_check_transforms_mode_all(monkeypatch, generation_output_dir_path):
    """
    Check transforms_mode 'all' for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "custom", "--stl-mesh-path", str(stl_file), "--transforms-mode", "all"]
    )
    args = parse_arguments()
    assert args.transforms_mode == 'all'


def test_custom_cli_check_transforms_mode_no_clipping(monkeypatch, generation_output_dir_path):
    """
    Check transforms_mode 'no-clipping' for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "custom", "--stl-mesh-path", str(stl_file), "--transforms-mode", "no-clipping"]
    )
    args = parse_arguments()
    assert args.transforms_mode == 'no-clipping'


def test_custom_cli_check_transforms_mode_invalid(monkeypatch, generation_output_dir_path):
    """
    Check invalid transforms_mode errors for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "custom", "--stl-mesh-path", str(stl_file), "--transforms-mode", "invalid"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_tissue_cli_check_transforms_mode_no_clipping(monkeypatch):
    """
    Check transforms_mode 'no-clipping' for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "tissue", "--transforms-mode", "no-clipping"]
    )
    args = parse_arguments()
    assert args.transforms_mode == 'no-clipping'


def test_tissue_cli_check_transforms_mode_invalid(monkeypatch):
    """
    Check invalid transforms_mode errors for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        ["script.py", "tissue", "--transforms-mode", "invalid"]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_tissue_cli_check_x_min_invalid_value(monkeypatch):
    """
    Case when we give an invalid x_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--x-min", "invalid"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_tissue_cli_check_default_x_min_value(monkeypatch):
    """
    Case when we check a default x_min value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.x_min == X_MIN


def test_tissue_cli_check_given_x_max_value(monkeypatch):
    """
    Case when we give an x_max argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--x-max", "10.0"
        ]
    )
    args = parse_arguments()
    assert args.x_max == 10.0


def test_tissue_cli_check_default_x_max_value(monkeypatch):
    """
    Case when we check a default x_max value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.x_max == X_MAX


def test_tissue_cli_check_given_y_min_value(monkeypatch):
    """
    Case when we give a y_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--y-min", "-10.0"
        ]
    )
    args = parse_arguments()
    assert args.y_min == -10.0


def test_tissue_cli_check_default_y_min_value(monkeypatch):
    """
    Case when we check a default y_min value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.y_min == Y_MIN


def test_tissue_cli_check_given_y_max_value(monkeypatch):
    """
    Case when we give a y_max argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--y-max", "10.0"
        ]
    )
    args = parse_arguments()
    assert args.y_max == 10.0


def test_tissue_cli_check_default_y_max_value(monkeypatch):
    """
    Case when we check a default y_max value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.y_max == Y_MAX


def test_tissue_cli_check_given_z_min_value(monkeypatch):
    """
    Case when we give a z_min argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--z-min", "-100.0"
        ]
    )
    args = parse_arguments()
    assert args.z_min == -100.0


def test_tissue_cli_check_default_z_min_value(monkeypatch):
    """
    Case when we check a default z_min value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.z_min == Z_MIN


def test_tissue_cli_check_given_z_max_value(monkeypatch):
    """
    Case when we give a z_max argument to tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue",
            "--z-max", "100.0"
        ]
    )
    args = parse_arguments()
    assert args.z_max == 100.0


def test_tissue_cli_check_default_z_max_value(monkeypatch):
    """
    Case when we check a default z_max value for tissue command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "tissue"
        ]
    )
    args = parse_arguments()
    assert args.z_max == Z_MAX


def test_custom_cli_check_given_stl_mesh_path_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a stl_mesh_path argument to custom command
    """
    stl_file = generation_output_dir_path / "test_mesh.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.stl_mesh_path == stl_file


def test_custom_cli_check_stl_mesh_path_missing(monkeypatch):
    """
    Case when we don't provide required stl_mesh_path argument to custom command
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_custom_cli_check_sample_children_only_inside_flag(monkeypatch, generation_output_dir_path):
    """
    Case when we set the sample_children_only_inside flag to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--sample-children-only-inside"
        ]
    )
    args = parse_arguments()
    assert args.sample_children_only_inside is True


def test_custom_cli_check_default_sample_children_only_inside_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check the default sample_children_only_inside value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.sample_children_only_inside == SAMPLE_CHILDREN_ONLY_INSIDE


def test_custom_cli_check_given_child_blobs_batch_size_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a child_blobs_batch_size argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--child-blobs-batch-size", "500000"
        ]
    )
    args = parse_arguments()
    assert args.child_blobs_batch_size == 500000


def test_custom_cli_check_child_blobs_batch_size_invalid_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give an invalid child_blobs_batch_size argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--child-blobs-batch-size", "invalid"
        ]
    )
    with pytest.raises(SystemExit):
        parse_arguments()


def test_custom_cli_check_default_child_blobs_batch_size_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check the default child_blobs_batch_size value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.child_blobs_batch_size == CHILD_BLOBS_BATCH_SIZE


def test_custom_cli_check_given_num_children_blobs_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a num_children_blobs argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--num-children-blobs", "7"
        ]
    )
    args = parse_arguments()
    assert args.num_children_blobs == 7


def test_custom_cli_check_default_num_children_blobs_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default num_children_blobs value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.num_children_blobs == NUM_CHILDREN_BLOBS


def test_custom_cli_check_given_density_min_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a density_min argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--density-min", "500.0"
        ]
    )
    args = parse_arguments()
    assert args.density_min == 500.0


def test_custom_cli_check_default_density_min_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default density_min value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.density_min == DENSITY_MIN


def test_custom_cli_check_given_density_max_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a density_max argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--density-max", "2500.0"
        ]
    )
    args = parse_arguments()
    assert args.density_max == 2500.0


def test_custom_cli_check_default_density_max_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default density_max value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.density_max == DENSITY_MAX


def test_custom_cli_check_given_conductivity_min_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a conductivity_min argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--conductivity-min", "0.5"
        ]
    )
    args = parse_arguments()
    assert args.conductivity_min == 0.5


def test_custom_cli_check_default_conductivity_min_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default conductivity_min value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.conductivity_min == CONDUCTIVITY_MIN


def test_custom_cli_check_given_conductivity_max_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a conductivity_max argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--conductivity-max", "3.0"
        ]
    )
    args = parse_arguments()
    assert args.conductivity_max == 3.0


def test_custom_cli_check_default_conductivity_max_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default conductivity_max value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.conductivity_max == CONDUCTIVITY_MAX


def test_custom_cli_check_given_permittivity_min_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a permittivity_min argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--permittivity-min", "5.0"
        ]
    )
    args = parse_arguments()
    assert args.permittivity_min == 5.0


def test_custom_cli_check_default_permittivity_min_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default permittivity_min value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.permittivity_min == PERMITTIVITY_MIN


def test_custom_cli_check_given_permittivity_max_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a permittivity_max argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--permittivity-max", "80.0"
        ]
    )
    args = parse_arguments()
    assert args.permittivity_max == 80.0


def test_custom_cli_check_default_permittivity_max_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default permittivity_max value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.permittivity_max == PERMITTIVITY_MAX


def test_custom_cli_check_given_blob_radius_decrease_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a blob_radius_decrease argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--blob-radius-decrease", "0.5"
        ]
    )
    args = parse_arguments()
    assert args.blob_radius_decrease == 0.5


def test_custom_cli_check_default_blob_radius_decrease_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default blob_radius_decrease value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.blob_radius_decrease == BLOB_RADIUS_DECREASE


def test_custom_cli_check_given_num_tubes_value(monkeypatch, generation_output_dir_path):
    """
    Case when we give a num_tubes argument to custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file),
            "--num-tubes", "15"
        ]
    )
    args = parse_arguments()
    assert args.num_tubes == 15


def test_custom_cli_check_default_num_tubes_value(monkeypatch, generation_output_dir_path):
    """
    Case when we check a default num_tubes value for custom command
    """
    stl_file = generation_output_dir_path / "test.stl"
    stl_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "script.py",
            "custom",
            "--stl-mesh-path", str(stl_file)
        ]
    )
    args = parse_arguments()
    assert args.num_tubes == NUM_TUBES
