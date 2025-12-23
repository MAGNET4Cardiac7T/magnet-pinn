"""Fixtures for property reader tests."""

import pytest
import numpy as np
import numpy.typing as npt
import pandas as pd
from shutil import rmtree
from trimesh import Trimesh

from magnet_pinn.preprocessing.reading_properties import (
    MATERIALS_FILE_NAME, FILE_COLUMN_NAME
)
from magnet_pinn.preprocessing.preprocessing import INPUT_DIR_PATH


@pytest.fixture(scope='module')
def property_data_invalid_columns(grid_simulation_path):
    """Create property data with invalid column names."""
    prop_path = grid_simulation_path / INPUT_DIR_PATH
    prop_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "invalid_column": [1, 2, 3],
            "another_invalid_column": [4, 5, 6]
        }
    )
    df.to_csv(prop_path / MATERIALS_FILE_NAME, index=False)

    yield prop_path
    if prop_path.exists():
        rmtree(prop_path)


@pytest.fixture(scope='module')
def property_data_invalid_file_name(grid_simulation_path):
    """Create property data with a non-existent file reference."""
    prop_path = grid_simulation_path / INPUT_DIR_PATH
    prop_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            FILE_COLUMN_NAME: ["file1"],
            "density": [1],
            "permittivity": [4],
            "conductivity": [7]
        }
    )
    df.to_csv(prop_path / MATERIALS_FILE_NAME, index=False)

    yield prop_path
    if prop_path.exists():
        rmtree(prop_path)


def create_mesh(cols: int, rows: int) -> Trimesh:
    """Create a simple planar mesh for testing."""
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)

    vertices = np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))

    faces_list: list[list[int]] = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces_list.append([idx, idx + 1, idx + cols + 1])
            faces_list.append([idx, idx + cols + 1, idx + cols])

    faces: npt.NDArray[np.int_] = np.array(faces_list)
    return Trimesh(vertices=vertices, faces=faces)


@pytest.fixture(scope='module')
def property_data_valid(grid_simulation_path):
    """Create valid property data with a mesh file."""
    prop_path = grid_simulation_path / INPUT_DIR_PATH
    prop_path.mkdir(parents=True, exist_ok=True)

    mesh_file_name = "mesh.stl"
    mesh = create_mesh(50, 50)
    mesh.export(prop_path / mesh_file_name)

    df = pd.DataFrame({
        FILE_COLUMN_NAME: [mesh_file_name],
        "density": [1],
        "permittivity": [4],
        "conductivity": [7]
    })
    df.to_csv(prop_path / MATERIALS_FILE_NAME, index=False)

    yield prop_path
    if prop_path.exists():
        rmtree(prop_path)
