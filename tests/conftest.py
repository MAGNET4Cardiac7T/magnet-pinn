import shutil
import pytest

import numpy as np
import numpy as np
from h5py import File

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    H_FIELD_DATABASE_KEY
)


@pytest.fixture(scope='session')
def grid_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    shutil.rmtree(simulation_path)


def create_grid_field(path: str, type: str, shape: tuple) -> None:
    """
    Shortcut to create a test .h5 file with a grid field.

    Due to that fact that original data was written in a wrong axis order, we 
    exchange x and z axis in the shape.

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
                np.zeros(shape=shape[::-1]),
                dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
            )
        )
        f.create_dataset("Mesh line x", data=np.zeros(shape[2]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.zeros(shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.zeros(shape[0]), dtype=np.float64)


@pytest.fixture(scope='session')
def e_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (126, 111, 121)
    )

    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (126, 111, 121)
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def h_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    create_grid_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (126, 111, 121)
    )

    create_grid_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (126, 111, 121)
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
