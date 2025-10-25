from pathlib import Path
from typing import Tuple, Callable, Union

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
from magnet_pinn.preprocessing.preprocessing import INPUT_DIR_PATH

INPUT_ANTENNA_DIR_PATH = "antenna"
RAW_DATA_DIR_PATH = "raw"
ANTENNA_SHORT_TERM_DIR_NAME = "antenna_short_term"
CENTRAL_BATCH_DIR_NAME = "central_batch"
CENTRAL_BATCH_SHORT_TERM_DIR_NAME = "central_batch_short_term"
SHIFTED_BATCH_DIR_NAME = "shifted_batch"
DUPLICATE_BATCH_DIR_NAME = "duplicate_batch"
CENTRAL_SPHERE_SIM_NAME = "children_0_tubes_0_id_0"
CENTRAL_BOX_SIM_NAME = "children_0_tubes_0_id_1"
SHIFTED_SPHERE_SIM_NAME = "children_0_tubes_0_id_2"
SHIFTED_BOX_SIM_NAME = "children_0_tubes_0_id_3"
FIELD_FILE_NAME_PATTERN = "{field}-field (f=297.2) [AC{fill_value}].h5"


def create_central_batch(data_dir_path, batch_dir_name: Union[str, Path] = CENTRAL_BATCH_DIR_NAME):
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / batch_dir_name

    create_simulation_data(batch_dir_path, CENTRAL_SPHERE_SIM_NAME)
    create_simulation_data(batch_dir_path, CENTRAL_BOX_SIM_NAME, create_box_input_data)

    return batch_dir_path


def create_shifted_batch(data_dir_path):
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / SHIFTED_BATCH_DIR_NAME

    create_simulation_data(batch_dir_path, SHIFTED_SPHERE_SIM_NAME, create_shifted_sphere_input_data)
    create_simulation_data(batch_dir_path, SHIFTED_BOX_SIM_NAME, create_shifted_box_input_data)

    return batch_dir_path


def create_duplicate_batch(data_dir_path):
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / DUPLICATE_BATCH_DIR_NAME

    create_simulation_data(batch_dir_path, CENTRAL_SPHERE_SIM_NAME)
    create_simulation_data(batch_dir_path, CENTRAL_BOX_SIM_NAME, create_box_input_data)

    return batch_dir_path


def create_antenna_test_data(data_path: Union[str, Path], antenna_dir_name: str = INPUT_ANTENNA_DIR_PATH) -> Path:
    """
    The method creates coils as boxes with their center of mass on 
    the X and Y axes correspondently. The are also symmetric to each
    others considering the point (0, 0, 0). It saves meshes to the antenna 
    directory. It also stores files names in the corresponding materials file 
    together with unit values of the features.
    """
    antenna_path = data_path / RAW_DATA_DIR_PATH / antenna_dir_name

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

    create_test_properties(antenna_path, coils_meshes)

    return antenna_path


def create_test_properties(prop_dir_path: str, meshes: Tuple[Trimesh]):
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


def create_sphere_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes = [
        Sphere(
            center=[0, 0, 0],
            radius=1
        )
    ]
    create_test_properties(input_dir_path, meshes)


def create_box_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes = [Box(bounds=np.array([
        [-1, -1, -1],
        [1, 1, 1]
    ]))]
    create_test_properties(input_dir_path, meshes)


def create_shifted_sphere_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes = [
        Sphere(
            center=[1, 1, 0],
            radius=1
        )
    ]
    create_test_properties(input_dir_path, meshes)


def create_shifted_box_input_data(simulation_dir_path: str):
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH
    meshes = [Box(bounds=np.array([
        [0, 0, -1],
        [2, 2, 1]
    ]))]
    create_test_properties(input_dir_path, meshes)


def create_simulation_data(simulations_dir_path: str, sim_name: str, subject_func: Callable = create_sphere_input_data):
    simulation_dir_path = simulations_dir_path / sim_name

    subject_func(simulation_dir_path)
    create_field(simulation_dir_path, E_FIELD_DATABASE_KEY, (9, 9, 9), 0)
    create_field(simulation_dir_path, E_FIELD_DATABASE_KEY, (9, 9, 9), 1)
    create_field(simulation_dir_path, E_FIELD_DATABASE_KEY, (9, 9, 9), 2)
    create_field(simulation_dir_path, E_FIELD_DATABASE_KEY, (9, 9, 9), 3)
    create_field(simulation_dir_path, H_FIELD_DATABASE_KEY, (9, 9, 9), 0)
    create_field(simulation_dir_path, H_FIELD_DATABASE_KEY, (9, 9, 9), 1)
    create_field(simulation_dir_path, H_FIELD_DATABASE_KEY, (9, 9, 9), 2)
    create_field(simulation_dir_path, H_FIELD_DATABASE_KEY, (9, 9, 9), 3)


def create_field(sim_path: str, field_type: str, shape: Tuple, fill_value: int = 0):
    """
    Creates an `.h5` field file with file name defined by `file_name`, 
    type of a field defined by `field_type`. The 
    """
    field_dir_path = sim_path / FIELD_DIR_PATH[field_type]
    field_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = field_dir_path / FIELD_FILE_NAME_PATTERN.format(field=field_type[0].lower(), fill_value=fill_value)
    bounds = np.array([[-4, -4, -4], [4, 4, 4]])

    create_grid_field(file_path, field_type, shape, bounds, fill_value)


def create_grid_field(file_path: str, field_type: str, shape: Tuple, bounds: npt.NDArray[np.float64], fill_value: int) -> None:
    """
    Shortcut to create a test .h5 file with a grid field.
    The field data creates a file with the field data which is a complex array with 0 imaginary part an `fill_value` as a real part.

    Parameters
    ----------
    path : str
        Path to the file
    type : str
        Type of the field
    shape : tuple
        Shape of the field
    """
    data = np.full(
        fill_value=fill_value,
        shape=shape,
        order='C',
        dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
    )
    data["x"]["im"] = 0
    data["y"]["im"] = 0
    data["z"]["im"] = 0
    with File(file_path, "w") as f:
        f.create_dataset(
            field_type,
            data=data
        )
        min_bounds = bounds[0]
        min_x, min_y, min_z = min_bounds
        max_bounds = bounds[1]
        max_x, max_y, max_z = max_bounds

        f.create_dataset("Mesh line x", data=np.linspace(min_x, max_x, shape[0]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.linspace(min_y, max_y, shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.linspace(min_z, max_z, shape[2]), dtype=np.float64)


def create_grid_field_with_mixed_axis_order(path: str, field_type: str, shape: Tuple, mixed_shape: Tuple) -> None:
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
            field_type,
            data=np.array(
                np.zeros(shape=mixed_shape),
                dtype=[('x', [('re', '<f4'), ('im', '<f4')]), ('y', [('re', '<f4'), ('im', '<f4')]), ('z', [('re', '<f4'), ('im', '<f4')])]
            )
        )
        f.create_dataset("Mesh line x", data=np.zeros(shape[0]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.zeros(shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.zeros(shape[2]), dtype=np.float64)


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
