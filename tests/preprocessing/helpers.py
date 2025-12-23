"""Helper functions for preprocessing tests."""

from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from h5py import File
from trimesh import Trimesh
from trimesh.primitives import Box, Sphere

from magnet_pinn.preprocessing.preprocessing import INPUT_DIR_PATH
from magnet_pinn.preprocessing.reading_field import (
    E_FIELD_DATABASE_KEY,
    FIELD_DIR_PATH,
    H_FIELD_DATABASE_KEY,
)
from magnet_pinn.preprocessing.reading_properties import (
    FEATURE_NAMES,
    FILE_COLUMN_NAME,
    MATERIALS_FILE_NAME,
)

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


def create_central_batch(
    data_dir_path: Path, batch_dir_name: Union[str, Path] = CENTRAL_BATCH_DIR_NAME
) -> Path:
    """Create a central batch directory with simulation data."""
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / batch_dir_name

    create_simulation_data(batch_dir_path, CENTRAL_SPHERE_SIM_NAME)
    create_simulation_data(batch_dir_path, CENTRAL_BOX_SIM_NAME, create_box_input_data)

    return batch_dir_path


def create_shifted_batch(data_dir_path: Path) -> Path:
    """Create a shifted batch directory with simulation data."""
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / SHIFTED_BATCH_DIR_NAME

    create_simulation_data(
        batch_dir_path, SHIFTED_SPHERE_SIM_NAME, create_shifted_sphere_input_data
    )
    create_simulation_data(
        batch_dir_path, SHIFTED_BOX_SIM_NAME, create_shifted_box_input_data
    )

    return batch_dir_path


def create_duplicate_batch(data_dir_path: Path) -> Path:
    """Create a duplicate batch directory with simulation data."""
    batch_dir_path = data_dir_path / RAW_DATA_DIR_PATH / DUPLICATE_BATCH_DIR_NAME

    create_simulation_data(batch_dir_path, CENTRAL_SPHERE_SIM_NAME)
    create_simulation_data(batch_dir_path, CENTRAL_BOX_SIM_NAME, create_box_input_data)

    return batch_dir_path


def create_antenna_test_data(
    data_path: Path, antenna_dir_name: str = INPUT_ANTENNA_DIR_PATH
) -> Path:
    """
    Create coils as boxes with their center of mass on the X and Y axes.

    The coils are also symmetric to each others considering the point (0, 0, 0).
    It saves meshes to the antenna directory. It also stores files names in the
    corresponding materials file together with unit values of the features.
    """
    antenna_path = data_path / RAW_DATA_DIR_PATH / antenna_dir_name

    coils_meshes: Sequence[Trimesh] = (
        Box(bounds=np.array([[2, -1, -1], [4, 1, 1]])),
        Box(bounds=np.array([[-4, -1, -1], [-2, 1, 1]])),
        Box(bounds=np.array([[-1, 2, -1], [1, 4, 1]])),
        Box(bounds=np.array([[-1, -4, -1], [1, -2, 1]])),
    )

    create_test_properties(antenna_path, coils_meshes)

    return antenna_path


def create_test_properties(prop_dir_path: Path, meshes: Sequence[Trimesh]) -> None:
    """Create a materials file with each property filled with 1."""
    prop_dir_path.mkdir(parents=True, exist_ok=True)

    mesh_names = []
    for i, mesh in enumerate(meshes):
        mesh_name = f"mesh_{i}.stl"
        mesh.export(prop_dir_path / mesh_name)
        mesh_names.append(mesh_name)

    df = pd.DataFrame(
        {
            FILE_COLUMN_NAME: mesh_names,
            **{name: np.ones(len(mesh_names)) for name in FEATURE_NAMES},
        }
    )
    df.to_csv(prop_dir_path / MATERIALS_FILE_NAME, index=False)


def create_sphere_input_data(simulation_dir_path: Path) -> None:
    """Create sphere input data for a simulation."""
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes: Sequence[Trimesh] = [Sphere(center=[0, 0, 0], radius=1)]
    create_test_properties(input_dir_path, meshes)


def create_box_input_data(simulation_dir_path: Path) -> None:
    """Create box input data for a simulation."""
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes: Sequence[Trimesh] = [Box(bounds=np.array([[-1, -1, -1], [1, 1, 1]]))]
    create_test_properties(input_dir_path, meshes)


def create_shifted_sphere_input_data(simulation_dir_path: Path) -> None:
    """Create shifted sphere input data for a simulation."""
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH

    meshes: Sequence[Trimesh] = [Sphere(center=[1, 1, 0], radius=1)]
    create_test_properties(input_dir_path, meshes)


def create_shifted_box_input_data(simulation_dir_path: Path) -> None:
    """Create shifted box input data for a simulation."""
    input_dir_path = simulation_dir_path / INPUT_DIR_PATH
    meshes: Sequence[Trimesh] = [Box(bounds=np.array([[0, 0, -1], [2, 2, 1]]))]
    create_test_properties(input_dir_path, meshes)


def create_simulation_data(
    simulations_dir_path: Path,
    sim_name: str,
    subject_func: Callable[[Path], None] = create_sphere_input_data,
) -> None:
    """Create simulation data with E and H fields."""
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


def create_field(
    sim_path: Path, field_type: str, shape: Tuple[int, ...], fill_value: int = 0
) -> None:
    """
    Create an .h5 field file.

    The file name is defined by file_name and type of a field is defined
    by field_type.
    """
    field_dir_path = sim_path / FIELD_DIR_PATH[field_type]
    field_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = field_dir_path / FIELD_FILE_NAME_PATTERN.format(
        field=field_type[0].lower(), fill_value=fill_value
    )
    bounds = np.array([[-4, -4, -4], [4, 4, 4]])

    create_grid_field(file_path, field_type, shape, bounds, fill_value)


def create_grid_field(
    file_path: Path,
    field_type: str,
    shape: Tuple[int, ...],
    bounds: npt.NDArray[np.float64],
    fill_value: int,
) -> None:
    """
    Create a test .h5 file with a grid field.

    The field data creates a file with the field data which is a complex array
    with 0 imaginary part and fill_value as a real part.

    Parameters
    ----------
    file_path : Path
        Path to the file.
    field_type : str
        Type of the field.
    shape : tuple
        Shape of the field.
    bounds : npt.NDArray[np.float64]
        Bounds of the field.
    fill_value : int
        Value to fill the field with.

    """
    field_dtype = [
        ("x", [("re", "<f4"), ("im", "<f4")]),
        ("y", [("re", "<f4"), ("im", "<f4")]),
        ("z", [("re", "<f4"), ("im", "<f4")]),
    ]
    data = np.full(fill_value=fill_value, shape=shape, order="C", dtype=field_dtype)
    data["x"]["im"] = 0
    data["y"]["im"] = 0
    data["z"]["im"] = 0
    with File(str(file_path), "w") as f:
        f.create_dataset(field_type, data=data)
        min_bounds = bounds[0]
        min_x, min_y, min_z = min_bounds
        max_bounds = bounds[1]
        max_x, max_y, max_z = max_bounds

        f.create_dataset(
            "Mesh line x", data=np.linspace(min_x, max_x, shape[0]), dtype=np.float64
        )
        f.create_dataset(
            "Mesh line y", data=np.linspace(min_y, max_y, shape[1]), dtype=np.float64
        )
        f.create_dataset(
            "Mesh line z", data=np.linspace(min_z, max_z, shape[2]), dtype=np.float64
        )


def create_grid_field_with_mixed_axis_order(
    path: Path, field_type: str, shape: Tuple[int, ...], mixed_shape: Tuple[int, ...]
) -> None:
    """
    Create a test .h5 file with a grid field with mixed axis order.

    Parameters
    ----------
    path : Path
        Path to the file.
    field_type : str
        Type of the field.
    shape : tuple
        Shape of the field.
    mixed_shape : tuple
        Mixed shape of the field.

    """
    field_dtype = [
        ("x", [("re", "<f4"), ("im", "<f4")]),
        ("y", [("re", "<f4"), ("im", "<f4")]),
        ("z", [("re", "<f4"), ("im", "<f4")]),
    ]
    with File(str(path), "w") as f:
        f.create_dataset(
            field_type, data=np.array(np.zeros(shape=mixed_shape), dtype=field_dtype)
        )
        f.create_dataset("Mesh line x", data=np.zeros(shape[0]), dtype=np.float64)
        f.create_dataset("Mesh line y", data=np.zeros(shape[1]), dtype=np.float64)
        f.create_dataset("Mesh line z", data=np.zeros(shape[2]), dtype=np.float64)


def create_pointslist_field(path: Path, field_type: str) -> None:
    """
    Create a test .h5 file with a pointslist field.

    Parameters
    ----------
    path : Path
        Path to the file.
    field_type : str
        Type of the field.

    """
    field_dtype = [
        ("x", [("re", "<f4"), ("im", "<f4")]),
        ("y", [("re", "<f4"), ("im", "<f4")]),
        ("z", [("re", "<f4"), ("im", "<f4")]),
    ]
    position_dtype = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    with File(str(path), "w") as f:
        f.create_dataset(
            field_type, data=np.array(np.zeros(100), dtype=field_dtype)
        )
        f.create_dataset(
            "Position", data=np.array(np.zeros(100), dtype=position_dtype)
        )
