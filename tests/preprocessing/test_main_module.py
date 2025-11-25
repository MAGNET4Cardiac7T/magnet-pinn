from argparse import Namespace
from pathlib import Path
import runpy
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest


def test_main_runs_grid_preprocessing(monkeypatch):
    fake_args = Namespace(
        preprocessing_type="grid",
        batches=[Path("batch_1")],
        antenna=Path("antenna_dir"),
        output=Path("output_dir"),
        field_dtype=np.float32,
        x_min=-1.0,
        x_max=1.0,
        y_min=-2.0,
        y_max=2.0,
        z_min=-3.0,
        z_max=3.0,
        voxel_size=4.0,
        simulations=[Path("sim_a"), Path("sim_b")]
    )
    grid_calls: dict[str, Any] = {}

    class DummyGrid:
        def __init__(
            self,
            batches,
            antenna,
            output,
            field_dtype,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            voxel_size
        ):
            grid_calls["init"] = dict(
                batches=batches,
                antenna=antenna,
                output=output,
                field_dtype=field_dtype,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
                voxel_size=voxel_size
            )
            grid_calls["instance"] = self

        def process_simulations(self, simulations):
            grid_calls["processed"] = simulations

    print_mock = Mock()

    monkeypatch.setattr("magnet_pinn.preprocessing.cli.parse_arguments", lambda: fake_args)
    monkeypatch.setattr("magnet_pinn.preprocessing.cli.print_report", print_mock)
    monkeypatch.setattr("magnet_pinn.preprocessing.preprocessing.GridPreprocessing", DummyGrid)

    result = runpy.run_module("magnet_pinn.preprocessing.__main__", run_name="magnet_pinn.preprocessing.__main__")

    assert grid_calls["init"] == dict(
        batches=fake_args.batches,
        antenna=fake_args.antenna,
        output=fake_args.output,
        field_dtype=fake_args.field_dtype,
        x_min=fake_args.x_min,
        x_max=fake_args.x_max,
        y_min=fake_args.y_min,
        y_max=fake_args.y_max,
        z_min=fake_args.z_min,
        z_max=fake_args.z_max,
        voxel_size=fake_args.voxel_size
    )
    assert grid_calls["processed"] == fake_args.simulations
    assert result["args"] is fake_args
    print_mock.assert_called_once_with(fake_args, grid_calls["instance"])


def test_main_runs_point_preprocessing(monkeypatch):
    fake_args = Namespace(
        preprocessing_type="point",
        batches=[Path("batch_2")],
        antenna=Path("antenna_dir"),
        output=Path("output_dir"),
        field_dtype=np.float64,
        simulations=None
    )
    point_calls: dict[str, Any] = {}

    class DummyPoint:
        def __init__(self, batches, antenna, output, field_dtype):
            point_calls["init"] = dict(
                batches=batches,
                antenna=antenna,
                output=output,
                field_dtype=field_dtype
            )
            point_calls["instance"] = self

        def process_simulations(self, simulations):
            point_calls["processed"] = simulations

    def _fail_grid(*args, **kwargs):
        raise AssertionError("GridPreprocessing should not be used for point preprocessing")

    print_mock = Mock()

    monkeypatch.setattr("magnet_pinn.preprocessing.cli.parse_arguments", lambda: fake_args)
    monkeypatch.setattr("magnet_pinn.preprocessing.cli.print_report", print_mock)
    monkeypatch.setattr("magnet_pinn.preprocessing.preprocessing.GridPreprocessing", _fail_grid)
    monkeypatch.setattr("magnet_pinn.preprocessing.preprocessing.PointPreprocessing", DummyPoint)

    result = runpy.run_module("magnet_pinn.preprocessing.__main__", run_name="magnet_pinn.preprocessing.__main__")

    assert point_calls["init"] == dict(
        batches=fake_args.batches,
        antenna=fake_args.antenna,
        output=fake_args.output,
        field_dtype=fake_args.field_dtype
    )
    assert point_calls["processed"] is None
    assert result["args"] is fake_args
    print_mock.assert_called_once_with(fake_args, point_calls["instance"])


def test_main_invalid_preprocessing_type(monkeypatch):
    fake_args = Namespace(
        preprocessing_type="unknown",
        batches=[Path("batch_3")],
        antenna=Path("antenna_dir"),
        output=Path("output_dir"),
        field_dtype=np.float32,
        simulations=[]
    )

    monkeypatch.setattr("magnet_pinn.preprocessing.cli.parse_arguments", lambda: fake_args)
    monkeypatch.setattr("magnet_pinn.preprocessing.preprocessing.GridPreprocessing", lambda *args, **kwargs: None)
    monkeypatch.setattr("magnet_pinn.preprocessing.preprocessing.PointPreprocessing", lambda *args, **kwargs: None)

    with pytest.raises(ValueError):
        runpy.run_module("magnet_pinn.preprocessing.__main__", run_name="magnet_pinn.preprocessing.__main__")
