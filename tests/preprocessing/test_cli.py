import pytest

from magnet_pinn.preprocessing.cli import (
    parse_arguments, BATCHES
)


def test_cli_check_grid_default_command(monkeypatch):
    """
    General check for the grid command
    """
    monkeypatch.setattr(
        "sys.argv", 
        [
            "script.py",
            "grid"
        ]
    )
    args = parse_arguments()
    assert args.preprocessing_type == "grid"


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


def test_cli_check_grid_batches_argmunets_for_one_value(monkeypatch, raw_central_batch_dir_path):
    """
    Case when we pass a grid command and one batch directory
    """
    monkeypatch.setattr(
        "sys.argv", 
        [
            "script.py",
            "grid",
            "--batches", str(raw_central_batch_dir_path)
        ]
    )
    args = parse_arguments()
    assert args.batches == [raw_central_batch_dir_path]


def test_cli_check_grid_batches_argmunets_for_multiple_values(monkeypatch, raw_central_batch_dir_path, raw_shifted_batch_dir_path):
    """
    Case when we pass a grid command and multiple batch directories
    """
    monkeypatch.setattr(
        "sys.argv", 
        [
            "script.py",
            "grid",
            "--batches", str(raw_shifted_batch_dir_path), str(raw_central_batch_dir_path)
        ]
    )
    args = parse_arguments()
    assert args.batches == [raw_shifted_batch_dir_path, raw_central_batch_dir_path]


def test_cli_check_grid_batches_arguments_default_value(monkeypatch):
    """
    Case when we pass a grid command and no batch directories
    """
    monkeypatch.setattr(
        "sys.argv", 
        [
            "script.py",
            "grid"
        ]
    )
    args = parse_arguments()
    assert args.batches == BATCHES
