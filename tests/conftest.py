import shutil
import pytest
from typing import Tuple

import numpy as np
import pandas as pd
from h5py import File
from trimesh import Trimesh

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    H_FIELD_DATABASE_KEY
)
from magnet_pinn.preprocessing.reading_properties import (
    MATERIALS_FILE_NAME, FILE_COLUMN_NAME
)
from magnet_pinn.preprocessing.preprocessing import INPUT_DIR_PATH


@pytest.fixture(scope='session')
def grid_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    shutil.rmtree(simulation_path)


def create_grid_field(path: str, type: str, shape: Tuple) -> None:
    """
    Shortcut to create a test .h5 file with a grid field.

    Parameters
    ----------
    path : str
        Path to the file
    type : str
        Type of the field
    shape : tuple
        Shape of the field
    """
    with File(path, "w") as f:
        f.create_dataset(
            type,
            data=np.array(
                np.zeros(shape=shape),
                dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
            )
        )
        f.create_dataset("Mesh line x", data=np.zeros(shape[0]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.zeros(shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.zeros(shape[2]), dtype=np.float64)


@pytest.fixture(scope='session')
def e_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126)
    )

    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126)
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def h_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126)
    )

    create_grid_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126)
    )

    return grid_simulation_path


def create_grid_field_with_mixed_axis_order(path: str, type: str, shape: Tuple, mixed_shape: Tuple) -> None:
    """
    Shortcut to create a test .h5 file with a grid field.

    Parameters
    ----------
    path : str
        Path to the file
    type : str
        Type of the field
    shape : tuple
        Shape of the field
    """
    with File(path, "w") as f:
        f.create_dataset(
            type,
            data=np.array(
                np.zeros(shape=mixed_shape),
                dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
            )
        )
        f.create_dataset("Mesh line x", data=np.zeros(shape[0]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.zeros(shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.zeros(shape[2]), dtype=np.float64)


@pytest.fixture(scope='session')
def e_field_grid_data_with_mixed_axis(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 121)
    )

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (126, 121, 111)
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def h_field_grid_data_with_mixed_axis(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 121)
    )

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (126, 121, 111)
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def e_field_grid_data_with_inconsistent_shape(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 126)
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def h_field_grid_data_with_inconsistent_shape(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field_with_mixed_axis_order(
        field_path / "h-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        (111, 126, 126)
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def pointslist_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    shutil.rmtree(simulation_path)


def create_pointslist_field(path: str, type: str) -> None:
    """
    Shortcut to create a test .h5 file with a pointslist field.

    Parameters
    ----------
    path : str
        Path to the file
    type : str
        Type of the field
    """
    with File(path, "w") as f:
        f.create_dataset(
            type,
            data=np.array(
                np.zeros(100),
                dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
            )
        )
        f.create_dataset(
            "Position", 
            data=np.array(
                np.zeros(100),
                dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')]
            )
        )


@pytest.fixture(scope='session')
def e_field_pointslist_data(pointslist_simulation_path):
    field_path = pointslist_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY
    )

    create_pointslist_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY
    )
    return pointslist_simulation_path


@pytest.fixture(scope='session')
def h_field_pointslist_data(pointslist_simulation_path):
    field_path = pointslist_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_pointslist_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY    
    )

    create_pointslist_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY
    )

    return pointslist_simulation_path


@pytest.fixture(scope='session')
def property_data_invalid_columns(grid_simulation_path):
    prop_path = grid_simulation_path / INPUT_DIR_PATH
    prop_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "invalid_column": [1, 2, 3],
            "another_invalid_column": [4, 5, 6]
        }
    )
    df.to_csv(prop_path / MATERIALS_FILE_NAME, index=False)

    return prop_path


@pytest.fixture(scope='session')
def property_data_invalid_file_name(grid_simulation_path):
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

    return prop_path


def create_mesh(cols: int, rows: int) -> Trimesh:
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)

    vertices = np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))

    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces.append([idx, idx + 1, idx + cols + 1])
            faces.append([idx, idx + cols + 1, idx + cols])

    faces = np.array(faces)
    return Trimesh(vertices=vertices, faces=faces)


@pytest.fixture(scope='session')
def property_data_valid(grid_simulation_path):
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

    return prop_path
