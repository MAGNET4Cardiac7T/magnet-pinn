import pytest
from pathlib import Path
from shutil import rmtree
from typing import Tuple, Callable

import numpy as np
import pandas as pd
from h5py import File
import numpy.typing as npt
from trimesh import Trimesh
from trimesh.primitives import Sphere, Box

from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    H_FIELD_DATABASE_KEY
)
from magnet_pinn.preprocessing.reading_properties import (
    MATERIALS_FILE_NAME, FILE_COLUMN_NAME, FEATURE_NAMES
)
from magnet_pinn.preprocessing.preprocessing import (
    INPUT_DIR_PATH, RAW_DATA_DIR_PATH, PROCESSED_DIR_PATH,
    INPUT_SIMULATIONS_DIR_PATH, INPUT_ANTENNA_DIR_PATH
)


BATCH_DIR_NAME = "batch"
CENTRAL_SPHERE_SIM_NAME = "children_0_tubes_0_id_0"
CENTRAL_BOX_SIM_NAME = "children_0_tubes_0_id_1"
SHIFTED_SPHERE_SIM_NAME = "children_0_tubes_0_id_2"
SHIFTED_BOX_SIM_NAME = "children_0_tubes_0_id_3"


@pytest.fixture(scope='session')
def data_dir_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp('data')
    yield data_path
    if data_path.exists():
        rmtree(data_path)


@pytest.fixture
def processed_batch_dir_path(data_dir_path):
    batch_path = data_dir_path / PROCESSED_DIR_PATH / BATCH_DIR_NAME
    batch_path.mkdir(parents=True, exist_ok=True)
    if batch_path.exists():
        rmtree(batch_path)


@pytest.fixture(scope='session')
def raw_batch_dir_path_long_term(data_dir_path):
    batch_dir_path = __create_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)


@pytest.fixture
def raw_batch_dir_path_short_term(data_dir_path):
    batch_dir_path = __create_batch(data_dir_path)
    yield batch_dir_path
    if batch_dir_path.exists():
        rmtree(batch_dir_path)

@pytest.fixture
def raw_antenna_dir_path(data_dir_path):
    antenna_path = __create_antenna_test_data(data_dir_path)
    yield antenna_path
    if antenna_path.exists():
        rmtree(antenna_path)
    

def __create_batch(data_dir_path):
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / BATCH_DIR_NAME

    __create_simulations(batch_dir_path)

    return batch_dir_path


def __create_antenna_test_data(data_path: str):
    """
    The method creates coils as boxes with their center of mass on 
    the X and Y axes correspondently. The are also symmetric to each
    others considering the point (0, 0, 0). It saves meshes to the antenna 
    directory. It also stores files names in the corresponding materials file 
    together with unit values of the features.
    """
    antenna_path = data_path / RAW_DATA_DIR_PATH / INPUT_ANTENNA_DIR_PATH

    coils_meshes = (
        Box(bounds=np.array([
            [2, -1, -1],
            [4, 1, 1]
        ])),
        Box(bounds=np.array([
            [-4, -1, -1],
            [-2, 1, 1]
        ])),
        Box(bounds=np.array([
            [-1, 2, -1],
            [1, 4, 1]
        ])),
        Box(bounds=np.array([
            [-1, -4, -1],
            [1, -2, 1]
        ]))
    )

    __create_test_properties(antenna_path, coils_meshes)

    return antenna_path


def __create_test_properties(prop_dir_path: str, meshes: Tuple[Trimesh]):
    """
    Creates a materials file with feeling each property just with 1.
    """
    prop_dir_path.mkdir(parents=True, exist_ok=True)
    
    mesh_names = []
    for i, mesh in enumerate(meshes):
        mesh_name = f"mesh_{i}.stl"
        mesh.export(prop_dir_path / mesh_name)
        mesh_names.append(mesh_name)
        

    df = pd.DataFrame(
        {
            FILE_COLUMN_NAME: mesh_names,
            **{name: np.ones(len(mesh_names)) for name in FEATURE_NAMES}
        }
    )
    df.to_csv(prop_dir_path / MATERIALS_FILE_NAME, index=False)


def __create_simulations(batch_dir_path: str):

    __create_simulation_data(batch_dir_path, CENTRAL_SPHERE_SIM_NAME)
    __create_simulation_data(batch_dir_path, CENTRAL_BOX_SIM_NAME, __create_box_input_data)
    __create_simulation_data(batch_dir_path, SHIFTED_SPHERE_SIM_NAME, __create_shifted_sphere_input_data)
    __create_simulation_data(batch_dir_path, SHIFTED_BOX_SIM_NAME, __create_shifted_box_input_data)

def __create_sphere_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes = [
        Sphere(
            center=[0, 0, 0],
            radius=1
        )
    ]
    __create_test_properties(input_dir_path, meshes)


def __create_box_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes = [Box(bounds=np.array([
        [-1, -1, -1],
        [1, 1, 1]
    ]))]
    __create_test_properties(input_dir_path, meshes)


def __create_shifted_sphere_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes = [
        Sphere(
            center=[1, 1, 0],
            radius=1
        )
    ]
    __create_test_properties(input_dir_path, meshes)


def __create_shifted_box_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH
    meshes = [Box(bounds=np.array([
        [0, 0, -1],
        [2, 2, 1]
    ]))]
    __create_test_properties(input_dir_path, meshes)


def __create_simulation_data(simulations_dir_path: str, sim_name: str, subject_func: Callable = __create_sphere_input_data):
    simulation_dir_path = simulations_dir_path / sim_name

    subject_func(simulation_dir_path)
    __create_field(simulation_dir_path, E_FIELD_DATABASE_KEY, (9, 9, 9))
    __create_field(simulation_dir_path, H_FIELD_DATABASE_KEY, (9, 9, 9))


def __create_field(sim_path: str, field_type: str, shape: Tuple, fill_value: float = 0):
    """
    Creates an `.h5` field file with file name defined by `file_name`, 
    type of a field defined by `field_type`. The 
    """
    field_dir_path = sim_path / FIELD_DIR_PATH[field_type]
    field_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = field_dir_path / "e-field (f=297.2) [AC1].h5"
    bounds = np.array([[-4, -4, -4], [4, 4, 4]])

    create_grid_field(file_path, field_type, shape, bounds)


@pytest.fixture(scope='session')
def processed_batch_dir_path(data_dir_path):
    batch_path = data_dir_path / PROCESSED_DIR_PATH / BATCH_DIR_NAME
    batch_path.mkdir(parents=True, exist_ok=True)
    return batch_path


@pytest.fixture(scope='session')
def grid_simulation_path(tmp_path_factory):
    simulation_path = tmp_path_factory.mktemp('simulation_name')
    yield simulation_path
    rmtree(simulation_path)


def create_grid_field(file_path: str, type: str, shape: Tuple, bounds: npt.NDArray[np.float_]) -> None:
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
    with File(file_path, "w") as f:
        f.create_dataset(
            type,
            data=np.array(
                np.zeros(shape=shape),
                dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
            )
        )
        min_bounds = bounds[0]
        min_x, min_y, min_z = min_bounds
        max_bounds = bounds[1]
        max_x, max_y, max_z = max_bounds

        f.create_dataset("Mesh line x", data=np.linspace(min_x, max_x, shape[0]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.linspace(min_y, max_y, shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.linspace(min_z, max_z, shape[2]), dtype=np.float64)


@pytest.fixture(scope='session')
def e_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[E_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-240, -220, -250], [240, 220, 250]])

    create_grid_field(
        field_path / "e-field (f=297.2) [AC1].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds
    )

    create_grid_field(
        field_path / "e-field (f=297.2) [AC2].h5",
        E_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds
    )

    return grid_simulation_path


@pytest.fixture(scope='session')
def h_field_grid_data(grid_simulation_path):
    field_path = grid_simulation_path / FIELD_DIR_PATH[H_FIELD_DATABASE_KEY]
    field_path.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[-240, -220, -250], [240, 220, 250]])

    create_grid_field(
        field_path / "h-field (f=297.2) [AC1].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds
    )

    create_grid_field(
        field_path / "h-field (f=297.2) [AC2].h5",
        H_FIELD_DATABASE_KEY,
        (121, 111, 126),
        bounds
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
    rmtree(simulation_path)


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
