from argparse import Namespace
from pathlib import Path
from typing import cast

from magnet_pinn.preprocessing.cli.helpers import print_report
from magnet_pinn.preprocessing.preprocessing import Preprocessing, GridPreprocessing


def test_print_report_without_grid_preprocessing(capsys):
    args = Namespace(
        batches=["b1", "b2"],
        simulations=["s1"],
        antenna="antenna_dir",
        output="output_dir",
        field_dtype="float32",
        preprocessing_type="point",
    )
    prep = cast(Preprocessing, Namespace(all_sim_paths=[Path("s1"), Path("s2"), Path("s3")]))

    print_report(args, prep)
    captured = capsys.readouterr().out

    assert "Preprocessing report:" in captured
    assert "Batches:  2" in captured
    assert "Overall simulations:  3" in captured
    assert "Chosen simulations:  1" in captured
    assert "Antenna:  antenna_dir" in captured
    assert "Output:  output_dir" in captured
    assert "Field data type:  float32" in captured
    assert "Preprocessing type:  point" in captured
    assert "x_min" not in captured


def test_print_report_with_grid_preprocessing(capsys):
    args = Namespace(
        batches=["batch_dir"],
        simulations=None,
        antenna="antenna_dir",
        output="output_dir",
        field_dtype="float64",
        preprocessing_type="grid",
        x_min=-1,
        x_max=1,
        y_min=-2,
        y_max=2,
        z_min=-3,
        z_max=3,
        voxel_size=5,
    )
    grid_prep = GridPreprocessing.__new__(GridPreprocessing)
    grid_prep.all_sim_paths = [Path("s1")]

    print_report(args, grid_prep)
    captured = capsys.readouterr().out

    assert "Chosen simulations:  All" in captured
    assert "Preprocessing type:  grid" in captured
    assert "x_min:  -1" in captured
    assert "x_max:  1" in captured
    assert "y_min:  -2" in captured
    assert "y_max:  2" in captured
    assert "z_min:  -3" in captured
    assert "z_max:  3" in captured
    assert "voxel size:  5" in captured
