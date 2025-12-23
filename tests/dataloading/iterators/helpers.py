"""Test helper functions for creating temporary processed data directories.

This module provides utilities to create temporary HDF5 file structures that
mimic the processed simulation data format used by the data iterators. These
helpers are used in test fixtures to set up grid and pointcloud test data
with proper directory structures, antenna coils, and simulation files.
"""
from pathlib import Path
from typing import Any, Dict, Callable

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
    """Create a complete processed data directory with antenna and simulation subdirectories.

    Creates the main directory and populates it with antenna coil data and two simulation
    files (one with random field values, one with zero field values) following the
    expected directory structure for data iterators.

    Parameters
    ----------
    grid_processed_dir : Path
        Path to the root directory to create.
    random_item : DataItem
        Data item with random field values for the first simulation.
    zero_item : DataItem
        Data item with zero field values for the second simulation.
    is_grid : bool
        If True, creates grid-specific attributes (voxel_size, extents).
        If False, creates pointcloud-specific structure.

    """
    grid_processed_dir.mkdir()

    create_grid_antenna_dir(grid_processed_dir, random_item)
    create_simulation_dir(grid_processed_dir, random_item, zero_item, is_grid)


def create_grid_antenna_dir(grid_processed_dir, random_grid_item):
    """Create antenna coils subdirectory with processed antenna masks file.

    Creates the antenna directory under the processed data root and populates it
    with an HDF5 file containing the antenna coil boolean masks.

    Parameters
    ----------
    grid_processed_dir : Path
        Root processed data directory path.
    random_grid_item : DataItem
        Data item containing coil masks to write to the antenna file.

    """
    grid_antenna_dir_path = grid_processed_dir / PROCESSED_ANTENNA_DIR_PATH
    grid_antenna_dir_path.mkdir()

    antenna_file_dir = grid_antenna_dir_path / TARGET_FILE_NAME.format(name="antenna")
    create_processed_coils_file(antenna_file_dir, random_grid_item)


def create_simulation_dir(grid_processed_dir, random_item, zero_item, is_grid: bool):
    """Create simulations subdirectory with two processed simulation HDF5 files.

    Creates the simulations directory under the processed data root and populates it
    with two simulation files - one with random field values and one with zero field
    values. Applies grid or pointcloud-specific attributes based on data type.

    Parameters
    ----------
    grid_processed_dir : Path
        Root processed data directory path.
    random_item : DataItem
        Data item with random field values for the first simulation file.
    zero_item : DataItem
        Data item with zero field values for the second simulation file.
    is_grid : bool
        If True, adds grid-specific attributes (voxel_size, extents).
        If False, adds pointcloud-specific attributes.

    """
    simulations_dir_path = grid_processed_dir / PROCESSED_SIMULATIONS_DIR_PATH
    simulations_dir_path.mkdir()

    additional_attributes = add_grid_attribures_to_file if is_grid else add_pointcloud_attributes_to_file

    random_grid_file_path = simulations_dir_path / TARGET_FILE_NAME.format(name=RANDOM_SIM_FILE_NAME)
    create_processed_simulation_file(random_grid_file_path, random_item, additional_attributes)

    zero_grid_file_path = simulations_dir_path / TARGET_FILE_NAME.format(name=ZERO_SIM_FILE_NAME)
    create_processed_simulation_file(zero_grid_file_path, zero_item, additional_attributes)


def create_processed_coils_file(path: Path, simulation: DataItem):
    """Create an HDF5 file with antenna coil boolean masks.

    Writes the antenna coil masks from the DataItem to an HDF5 file using the
    standard ANTENNA_MASKS_OUT_KEY dataset name and boolean dtype.

    Parameters
    ----------
    path : Path
        File path where the HDF5 coils file will be created.
    simulation : DataItem
        Data item containing the coil masks to write. Must have non-None coils attribute.

    """
    with File(str(path), "w") as f:
        assert simulation.coils is not None
        f.create_dataset(ANTENNA_MASKS_OUT_KEY, data=simulation.coils.astype(np.bool_), dtype=np.bool_)


def create_processed_simulation_file(
    path: Path, simulation: DataItem, additional_attributes: Callable
):
    """Create an HDF5 simulation file with electromagnetic fields and metadata.

    Writes E-field, H-field, features (input), subject masks, and metadata attributes
    to an HDF5 file. The file format is shape-agnostic and works for both grid and
    pointcloud data. Grid/pointcloud-specific attributes are added via the
    additional_attributes callback.

    Parameters
    ----------
    path : Path
        File path where the HDF5 simulation file will be created.
    simulation : DataItem
        Data item containing simulation data. Must have non-None field, input, subject,
        dtype, and truncation_coefficients attributes.
    additional_attributes : Callable
        Callback function to add data type-specific attributes (grid: voxel_size/extents,
        pointcloud: coordinates only). Signature: (h5py.File, DataItem) -> None.

    """
    assert simulation.dtype is not None
    efield, other_prop_field = format_field(simulation.field[0], simulation.dtype)
    hfield, _ = format_field(simulation.field[1], simulation.dtype)

    with File(str(path), "w") as f:
        f.create_dataset(E_FIELD_OUT_KEY, data=efield)
        f.create_dataset(H_FIELD_OUT_KEY, data=hfield)
        f.create_dataset(FEATURES_OUT_KEY, data=simulation.input.astype(other_prop_field))
        f.create_dataset(SUBJECT_OUT_KEY, data=simulation.subject.astype(np.bool_))

        f.attrs[DTYPE_OUT_KEY] = simulation.dtype
        assert simulation.truncation_coefficients is not None
        f.attrs[TRUNCATION_COEFFICIENTS_OUT_KEY] = simulation.truncation_coefficients.astype(other_prop_field)

        additional_attributes(f, simulation)


def add_grid_attribures_to_file(f: File, simulation: DataItem):
    """Add grid-specific attributes and coordinates to an HDF5 simulation file.

    Writes voxel size, spatial extents (min/max), and coordinate positions as
    attributes and datasets in the HDF5 file. Uses test-specific hardcoded values
    for voxel size (4) and extents (-40 to 36 in each dimension).

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle to write attributes and datasets to.
    simulation : DataItem
        Data item containing position coordinates to write as the coordinates dataset.

    """
    f.attrs[VOXEL_SIZE_OUT_KEY] = 4
    f.attrs[MIN_EXTENT_OUT_KEY] = np.array([-40, -40, -40])
    f.attrs[MAX_EXTENT_OUT_KEY] = np.array([36, 36, 36])
    f.create_dataset(COORDINATES_OUT_KEY, data=simulation.positions)


def add_pointcloud_attributes_to_file(f: File, simulation: DataItem):
    """Add pointcloud-specific coordinates dataset to an HDF5 simulation file.

    Writes the point positions as the coordinates dataset in the HDF5 file.
    Unlike grid data, pointcloud data only requires coordinates without
    voxel size or extent attributes.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle to write the dataset to.
    simulation : DataItem
        Data item containing point positions to write as the coordinates dataset.

    """
    f.create_dataset(COORDINATES_OUT_KEY, data=simulation.positions)


def format_field(
    field: np.ndarray, dtype: str
) -> tuple[np.ndarray, np.dtype[np.floating[Any]]]:
    """Format electromagnetic field from real/imaginary arrays to storage format.

    Converts field data from separate real/imaginary arrays to either structured
    float arrays (with 're' and 'im' fields) or complex arrays, depending on the
    target dtype. Also determines the appropriate dtype for non-field data.

    Parameters
    ----------
    field : np.ndarray
        Field array with shape (2, coils, components, spatial...) where field[0]
        is real part and field[1] is imaginary part.
    dtype : str
        Target dtype string (e.g., 'float32', 'complex64').

    Returns
    -------
    result_field : np.ndarray
        Formatted field array either as structured dtype with 're'/'im' fields
        (for float dtypes) or as complex array (for complex dtypes).
    other_types : np.dtype
        Appropriate dtype for non-field data (features, truncation coefficients).
        For float dtypes, returns the same dtype. For complex dtypes, returns
        the corresponding float dtype (e.g., complex64 -> float32).

    """
    writing_type = np.dtype(dtype)
    real = field[0]  # Shape: (coils, components, spatial...)
    im = field[1]    # Shape: (coils, components, spatial...)
    if writing_type.kind == FLOAT_DTYPE_KIND:
        field_type = [("re", writing_type), ("im", writing_type)]
        other_types = writing_type
        result_field = np.empty_like(real, dtype=field_type)
        result_field["re"] = real
        result_field["im"] = im
    elif writing_type.kind == COMPLEX_DTYPE_KIND:
        # field_type = writing_type  # Dead assignment, removed
        float_size = 8 * writing_type.itemsize // 2
        other_types = np.dtype(f"float{float_size}")
        result_field = real + 1j * im

    return result_field, other_types


def check_dtypes_between_iter_result_and_supposed_simulation(result: Dict, item: DataItem):
    """Validate that iterator output dtypes match expected dtypes from reference item.

    Compares all data component dtypes between the iterator result dictionary and
    a reference DataItem to ensure the iterator is loading and preserving dtypes
    correctly from HDF5 files.

    Parameters
    ----------
    result : Dict
        Dictionary returned by the iterator containing data components
        (simulation, input, field, subject, positions, phase, mask, coils,
        dtype, truncation_coefficients).
    item : DataItem
        Reference data item with expected dtypes. All relevant attributes
        (input, field, subject, etc.) must be non-None.

    """
    # Type narrowing assertions
    assert item.positions is not None
    assert item.phase is not None
    assert item.mask is not None
    assert item.coils is not None
    assert item.truncation_coefficients is not None

    assert isinstance(item.simulation, type(result["simulation"]))
    assert item.input.dtype == result["input"].dtype
    assert item.field.dtype == result["field"].dtype
    assert item.subject.dtype == result["subject"].dtype
    assert item.positions.dtype == result["positions"].dtype
    assert item.phase.dtype == result["phase"].dtype
    assert item.mask.dtype == result["mask"].dtype
    assert item.coils.dtype == result["coils"].dtype
    assert isinstance(item.dtype, type(result["dtype"]))
    assert (
        item.truncation_coefficients.dtype
        == result["truncation_coefficients"].dtype
    )


def check_shapes_between_item_result_and_supposed_simulation(result: Dict, item: DataItem):
    """Validate that iterator output shapes match expected shapes from reference item.

    Compares all data component shapes between the iterator result dictionary and
    a reference DataItem to ensure the iterator is loading data with correct dimensions.
    Accounts for transform-induced shape changes (e.g., field losing one spatial dim,
    subject reducing from (1, points) to (points,), coils gaining real/imag dimension).

    Parameters
    ----------
    result : Dict
        Dictionary returned by the iterator containing data components.
    item : DataItem
        Reference data item with expected shapes before iterator transforms.
        All relevant attributes must be non-None.

    """
    # Type narrowing assertions
    assert item.positions is not None
    assert item.phase is not None
    assert item.mask is not None
    assert item.coils is not None
    assert item.dtype is not None
    assert item.truncation_coefficients is not None

    assert item.input.shape == result["input"].shape
    expected_field_shape = (
        item.field.shape[0],
        item.field.shape[1],
        *item.field.shape[3:],
    )
    assert expected_field_shape == result["field"].shape
    assert item.subject.shape[1:] == result["subject"].shape
    assert item.positions.shape == result["positions"].shape
    assert item.phase.shape == result["phase"].shape
    assert item.mask.shape == result["mask"].shape
    assert tuple([2] + list(item.coils.shape[1:])) == result["coils"].shape
    assert len(item.dtype) == len(result["dtype"])
    assert (
        item.truncation_coefficients.shape
        == result["truncation_coefficients"].shape
    )


def check_shapes_between_item_result_and_supposed_simulation_for_pointclous(result: Dict, item: DataItem):
    """Validate iterator output shapes for pointcloud data after PointPhaseShift transform.

    Similar to check_shapes_between_item_result_and_supposed_simulation but specifically
    for pointcloud data. Verifies that all data component shapes match expectations
    after the PointPhaseShift transform is applied, including subject dimension reduction
    from (1, points) to (points,) via np.max(axis=0).

    Parameters
    ----------
    result : Dict
        Dictionary returned by the pointcloud iterator containing data components.
    item : DataItem
        Reference pointcloud data item with expected shapes before iterator transforms.
        All relevant attributes must be non-None.

    """
    # Type narrowing assertions
    assert item.positions is not None
    assert item.phase is not None
    assert item.mask is not None
    assert item.coils is not None
    assert item.dtype is not None
    assert item.truncation_coefficients is not None

    assert item.input.shape == result["input"].shape
    # Iterator reduces subject from (1, points) to (points,) via np.max(axis=0)
    assert item.subject.shape[1:] == result["subject"].shape
    assert item.positions.shape == result["positions"].shape
    assert item.phase.shape == result["phase"].shape
    assert item.mask.shape == result["mask"].shape
    assert len(item.dtype) == len(result["dtype"])
    assert item.truncation_coefficients.shape == result["truncation_coefficients"].shape

    expected_field_shape = (item.field.shape[0], item.field.shape[1], *item.field.shape[3:])
    assert expected_field_shape == result["field"].shape
    assert tuple([2] + list(item.coils.shape[1:])) == result["coils"].shape


def check_values_between_item_result_and_supposed_simulation(result: Dict, item: DataItem):
    """Validate that iterator output values match expected values from reference item.

    Checks that non-transformed data components (simulation ID, input, positions,
    dtype, truncation coefficients) have identical values between the iterator
    result and reference item. Also verifies that transformed components (phase,
    mask) have different values as expected from the PhaseShift transform.

    Parameters
    ----------
    result : Dict
        Dictionary returned by the iterator containing data components.
    item : DataItem
        Reference data item with expected values. All relevant attributes
        (input, positions, phase, mask, truncation_coefficients) must be non-None.

    """
    # Type narrowing assertions
    assert item.positions is not None
    assert item.phase is not None
    assert item.mask is not None
    assert item.truncation_coefficients is not None

    print(item.simulation, result["simulation"])
    assert item.simulation == result["simulation"]
    assert np.equal(item.input, result["input"]).all()
    assert np.equal(item.positions, result["positions"]).all()
    assert not np.equal(item.phase, result["phase"]).all()
    assert not np.equal(item.mask, result["mask"]).all()
    assert item.dtype == result["dtype"]
    assert np.equal(item.truncation_coefficients, result["truncation_coefficients"]).all()
