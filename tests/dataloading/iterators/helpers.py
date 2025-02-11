from pathlib import Path
from typing import Dict, Callable

import numpy as np
from h5py import File

from magnet_pinn.data.dataitem import DataItem
from magnet_pinn.preprocessing.preprocessing import (
    ANTENNA_MASKS_OUT_KEY, E_FIELD_OUT_KEY, H_FIELD_OUT_KEY,
    FEATURES_OUT_KEY, SUBJECT_OUT_KEY, FLOAT_DTYPE_KIND, COMPLEX_DTYPE_KIND,
    DTYPE_OUT_KEY, TRUNCATION_COEFFICIENTS_OUT_KEY, VOXEL_SIZE_OUT_KEY,
    MIN_EXTENT_OUT_KEY, MAX_EXTENT_OUT_KEY, PROCESSED_SIMULATIONS_DIR_PATH,
    TARGET_FILE_NAME, PROCESSED_ANTENNA_DIR_PATH, COORDINATES_OUT_KEY
)


RANDOM_SIM_FILE_NAME = "children_0_tubes_0_id_0"
ZERO_SIM_FILE_NAME = "children_0_tubes_0_id_1"


def create_processed_dir(grid_processed_dir, random_item, zero_item, is_grid: bool):
    grid_processed_dir.mkdir()

    create_grid_antenna_dir(grid_processed_dir, random_item)
    create_simulation_dir(grid_processed_dir, random_item, zero_item, is_grid)


def create_grid_antenna_dir(grid_processed_dir, random_grid_item):
    grid_antenna_dir_path = grid_processed_dir / PROCESSED_ANTENNA_DIR_PATH
    grid_antenna_dir_path.mkdir()

    antenna_file_dir = grid_antenna_dir_path / TARGET_FILE_NAME.format(name="antenna")
    create_processed_coils_file(antenna_file_dir, random_grid_item)


def create_simulation_dir(grid_processed_dir, random_item, zero_item, is_grid: bool):
    simulations_dir_path = grid_processed_dir / PROCESSED_SIMULATIONS_DIR_PATH
    simulations_dir_path.mkdir()

    additional_attributes = add_grid_attribures_to_file if is_grid else add_pointcloud_attributes_to_file

    random_grid_file_path = simulations_dir_path / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME)
    create_processed_simulation_file(random_grid_file_path, random_item, additional_attributes)

    zero_grid_file_path = simulations_dir_path / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME)
    create_processed_simulation_file(zero_grid_file_path, zero_item, additional_attributes)


def create_processed_coils_file(path: Path, simulation: DataItem):
    with File(path, "w") as f:
        f.create_dataset(ANTENNA_MASKS_OUT_KEY, data=simulation.coils.astype(np.bool_), dtype=np.bool_)


def create_processed_simulation_file(path: Path, simulation: DataItem, additional_attributes: Callable):
    """
    Considering that fact that grid/points datasets writing is shape-agnostic, we can unify it, just add additional functions
    to write specific attributes
    """
    efield, other_prop_field = format_field(simulation.field[0], simulation.dtype)
    hfied, _ = format_field(simulation.field[1], simulation.dtype)

    with File(path, "w") as f:
        f.create_dataset(E_FIELD_OUT_KEY, data=efield)
        f.create_dataset(H_FIELD_OUT_KEY, data=hfied)
        f.create_dataset(FEATURES_OUT_KEY, data=simulation.input.astype(other_prop_field))
        f.create_dataset(SUBJECT_OUT_KEY, data=simulation.subject.astype(np.bool_))

        f.attrs[DTYPE_OUT_KEY] = simulation.dtype
        f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY] = simulation.truncation_coefficients.astype(other_prop_field)

        additional_attributes(f, simulation)


def add_grid_attribures_to_file(f: File, simulation: DataItem):
    f.attrs[VOXEL_SIZE_OUT_KEY] = 4
    f.attrs[MIN_EXTENT_OUT_KEY] = np.array([-40, -40, -40])
    f.attrs[MAX_EXTENT_OUT_KEY] = np.array([36, 36, 36])


def add_pointcloud_attributes_to_file(f: File, simulation: DataItem):
    f.create_dataset(COORDINATES_OUT_KEY, data=simulation.positions.astype(np.float32))


def format_field(field: np.ndarray, dtype: str) -> np.ndarray:
    """
    Method formats e/h field values and also returns the dtype of all other values
    """
    writing_type = np.dtype(dtype)
    real = field[0]
    im = field[1]
    if writing_type.kind == FLOAT_DTYPE_KIND:
        field_type = [("re", writing_type),("im", writing_type)]
        other_types = writing_type
        result_field = np.empty_like(real, dtype=field_type)
        result_field["re"] = real
        result_field["im"] = im
    elif writing_type.kind == COMPLEX_DTYPE_KIND:
        field_type = writing_type
        float_size = 8 * writing_type.itemsize // 2
        other_types = np.dtype(f"float{float_size}")
        result_field = real + 1j * im

    return result_field, other_types


def check_dtypes_between_iter_result_and_supposed_simulation(result: Dict, item: DataItem):
    assert type(item.simulation) == type(result["simulation"])
    assert item.input.dtype == result["input"].dtype
    assert item.field.dtype == result["field"].dtype
    assert item.subject.dtype == result["subject"].dtype
    assert item.positions.dtype == result["positions"].dtype
    assert item.phase.dtype == result["phase"].dtype
    assert item.mask.dtype == result["mask"].dtype
    assert item.coils.dtype == result["coils"].dtype
    assert type(item.dtype) == type(result["dtype"])
    assert item.truncation_coefficients.dtype == result["truncation_coefficients"].dtype


def check_shapes_between_item_result_and_supposed_simulation(result: Dict, item: DataItem):
    assert item.input.shape == result["input"].shape
    assert item.field.shape[:-1] == result["field"].shape
    assert item.subject.shape[:-1] == result["subject"].shape
    assert item.positions.shape == result["positions"].shape
    assert item.phase.shape == result["phase"].shape
    assert item.mask.shape == result["mask"].shape
    assert tuple([2] + list(item.coils.shape[:-1])) == result["coils"].shape
    assert len(item.dtype) == len(result["dtype"])
    assert item.truncation_coefficients.shape == result["truncation_coefficients"].shape


def check_shapes_between_item_result_and_supposed_simulation_for_pointclous(result: Dict, item: DataItem):
    """
    During the testing we gonna use compose of `PointPhaseShift` and `PointFeatureRearrange` transforms so our dimensions will be mirrowed
    for the field and coils properties.
    """
    assert item.input.shape == result["input"].shape
    assert item.subject.shape[:-1] == result["subject"].shape
    assert item.positions.shape == result["positions"].shape
    assert item.phase.shape == result["phase"].shape
    assert item.mask.shape == result["mask"].shape
    assert len(item.dtype) == len(result["dtype"])
    assert item.truncation_coefficients.shape == result["truncation_coefficients"].shape

    assert tuple([item.field.shape[2], item.field.shape[-2], item.field.shape[1], item.field.shape[0]]) == result["field"].shape
    assert tuple([item.coils.shape[0], 2]) == result["coils"].shape


def check_values_between_item_result_and_supposed_simulation(result: Dict, item: DataItem):
    """
    We check all elements of the item which potentially have same shape and not changed
    """
    print(item.simulation, result["simulation"])
    assert item.simulation == result["simulation"]
    assert np.equal(item.input, result["input"]).all()
    assert np.equal(item.positions, result["positions"]).all()
    assert not np.equal(item.phase, result["phase"]).all()
    assert not np.equal(item.mask, result["mask"]).all()
    assert item.dtype == result["dtype"]
    assert np.equal(item.truncation_coefficients, result["truncation_coefficients"]).all()
