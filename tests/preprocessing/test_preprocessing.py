from os import listdir
from pathlib import Path
from shutil import rmtree

import pytest
import numpy as np
from h5py import File
from natsort import natsorted

from tests.preprocessing.conftest import ALL_SIM_NAMES
from tests.preprocessing.helpers import (
    CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME,
    SHIFTED_BOX_SIM_NAME, SHIFTED_SPHERE_SIM_NAME
)
from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PointPreprocessing, 
    PROCESSED_ANTENNA_DIR_PATH, PROCESSED_SIMULATIONS_DIR_PATH, TARGET_FILE_NAME,
    ANTENNA_MASKS_OUT_KEY, E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, 
    SUBJECT_OUT_KEY, COORDINATES_OUT_KEY, DTYPE_OUT_KEY,
    TRUNCATION_COEFFICIENTS_OUT_KEY, COORDINATES_OUT_KEY,
    MIN_EXTENT_OUT_KEY, MAX_EXTENT_OUT_KEY, VOXEL_SIZE_OUT_KEY
)


def test_grid_out_dir_structure(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    p.process_simulations()

    case_name = "grid_voxel_size_1_data_type_complex64"
    expected_batch_case_dir = processed_batch_dir_path / case_name

    assert expected_batch_case_dir == p.out_antenna_dir_path.parent
    assert expected_batch_case_dir == p.out_simulations_dir_path.parent
    assert expected_batch_case_dir.exists()

    expected_antenna_dir = expected_batch_case_dir / PROCESSED_ANTENNA_DIR_PATH
    assert p.out_antenna_dir_path == expected_antenna_dir
    assert expected_antenna_dir.exists()

    expected_simulations_dir = expected_batch_case_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert p.out_simulations_dir_path == expected_simulations_dir
    assert expected_simulations_dir.exists()

    for simulation_dir in expected_simulations_dir.iterdir():
        assert simulation_dir.is_file()


def test_grid_antenna(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    p.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_antenna_file = Path(p.out_antenna_dir_path) / TARGET_FILE_NAME.format(name="antenna")
    assert out_antenna_file.exists()

    with File(out_antenna_file) as f:
        assert list(f.keys()) == [ANTENNA_MASKS_OUT_KEY]
        assert list(f.attrs.keys()) == []
        masks = f[ANTENNA_MASKS_OUT_KEY][:]
        assert masks.shape == (4, 9, 9, 9)
        assert masks.dtype == np.bool_

        received_mask = np.sum(masks[:, :, :, 3], axis=0)
        expected_mask = np.zeros((9, 9))
        expected_mask[3:6, 0:3] = 1
        expected_mask[3:6, 6:9] = 1
        expected_mask[0:3, 3:6] = 1
        expected_mask[6:9, 3:6] = 1
        assert np.equal(received_mask, expected_mask).all()


def test_grid_out_simulation_structure(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    p = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    p.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = p.out_simulations_dir_path /TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)

    with File(sim_file) as f:
        assert set(f.keys()) == {E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY, COORDINATES_OUT_KEY}
        assert set(f.attrs.keys()) == {DTYPE_OUT_KEY, TRUNCATION_COEFFICIENTS_OUT_KEY, MIN_EXTENT_OUT_KEY, MAX_EXTENT_OUT_KEY, VOXEL_SIZE_OUT_KEY}


def test_grid_central_complex_one_simulation_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        [CENTRAL_SPHERE_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    assert len(list(listdir(out_simulations_dir))) == 1

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_complex_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def test_grid_central_complex_multiple_simulations_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        [CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    assert len(list(listdir(out_simulations_dir))) == 2

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_complex_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)

    next_sim_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    assert next_sim_file.exists()

    with File(next_sim_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_complex_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def check_antenna(out_dir: str):
    out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
    assert out_antenna_dir.exists()

    out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
    assert out_antenna_file.exists()

    with File(out_antenna_file) as f:
        assert list(f.keys()) == [ANTENNA_MASKS_OUT_KEY]
        masks = f[ANTENNA_MASKS_OUT_KEY][:]
        assert masks.shape == (4, 9, 9, 9)
        assert masks.dtype == np.bool_

        received_mask = np.sum(masks[:, :, :, 3], axis=0)
        expected_mask = np.zeros((9, 9))
        expected_mask[3:6, 0:3] = 1
        expected_mask[3:6, 6:9] = 1
        expected_mask[0:3, 3:6] = 1
        expected_mask[6:9, 3:6] = 1
        assert np.equal(received_mask, expected_mask).all()


def test_grid_central_float_one_simulation_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        [CENTRAL_SPHERE_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_float32"
    out_dir = processed_batch_dir_path / case_name
    assert out_dir.exists()

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    assert len(list(listdir(out_simulations_dir))) == 1

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_float_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def test_grid_central_float_multiple_simulations_valid_preprocessing(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        [CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_float32"
    out_dir = processed_batch_dir_path / case_name
    assert out_dir.exists()

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    assert len(list(listdir(out_simulations_dir))) == 2

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_float_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)

    another_sim_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    assert another_sim_file.exists()

    with File(another_sim_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_float_fields(f)
        check_central_features(f)
        check_central_subject_mask(f)


def test_grid_shifted_one_simulation_valid_preprocessing(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        [SHIFTED_SPHERE_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    assert out_simulation_file.exists()

    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]
        check_shifted_subject_mask(f)
        check_shifted_features(f)


def check_complex_fields(f: File):
    expected_field = np.concatenate(
        [
            np.zeros((3, 9, 9, 9, 1), dtype=np.complex64), 
            np.ones((3, 9, 9, 9, 1), dtype=np.complex64),
            np.full(fill_value=2, shape=(3, 9, 9, 9, 1), dtype=np.complex64)
        ],
        axis=-1
    )

    e_field = f[E_FIELD_OUT_KEY][:]
    assert e_field.shape == (3, 9, 9, 9, 3)
    assert e_field.dtype == np.complex64
    assert np.equal(e_field, expected_field).all()

    h_field = f[H_FIELD_OUT_KEY][:]
    assert h_field.shape == (3, 9, 9, 9, 3)
    assert h_field.dtype == np.complex64
    assert np.equal(h_field, expected_field).all()


def check_float_fields(f: File):
    expected_re_field = np.concatenate(
        [
            np.zeros((3, 9, 9, 9, 1), dtype=np.float32), 
            np.ones((3, 9, 9, 9, 1), dtype=np.float32),
            np.full(fill_value=2, shape=(3, 9, 9, 9, 1), dtype=np.float32)
        ],
        axis=-1
    )

    expected_im_field = np.zeros((3, 9, 9, 9, 3), dtype=np.float32)

    e_field = f[E_FIELD_OUT_KEY][:]
    assert e_field.shape == (3, 9, 9, 9, 3)
    assert e_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
    re_e_field = e_field["re"]
    assert np.equal(re_e_field, expected_re_field).all()
    im_e_field = e_field["im"]
    assert np.equal(im_e_field, expected_im_field).all()

    h_field = f[H_FIELD_OUT_KEY][:]
    assert h_field.shape == (3, 9, 9, 9, 3)
    assert h_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
    re_h_field = h_field["re"]
    assert np.equal(re_h_field, expected_re_field).all()
    im_h_field = h_field["im"]
    assert np.equal(im_h_field, expected_im_field).all()


def check_central_subject_mask(f: File):
    subject_mask = f[SUBJECT_OUT_KEY][:]
    assert subject_mask.shape == (1, 9, 9, 9)
    assert subject_mask.dtype == np.bool_
    expected_subject_mask = np.zeros((9, 9), dtype=np.bool_)
    expected_subject_mask[4, 4] = 1
    empty_mask = np.zeros((9, 9), dtype=np.bool_)
    assert np.equal(subject_mask[0, :, :, 3], empty_mask).all()
    assert np.equal(subject_mask[0, :, :, 4], expected_subject_mask).all()
    assert np.equal(subject_mask[0, :, :, 5], empty_mask).all()

def check_shifted_subject_mask(f: File):
    subject_mask = f[SUBJECT_OUT_KEY][:]
    assert subject_mask.shape == (1, 9, 9, 9)
    assert subject_mask.dtype == np.bool_
    expected_subject_mask = np.zeros((9, 9), dtype=np.bool_)
    expected_subject_mask[5, 5] = 1
    empty_mask = np.zeros((9, 9), dtype=np.bool_)
    assert np.equal(subject_mask[0, :, :, 3], empty_mask).all()
    assert np.equal(subject_mask[0, :, :, 4], expected_subject_mask).all()
    assert np.equal(subject_mask[0, :, :, 5], empty_mask).all()


def check_central_features(f: File):
    features = f[FEATURES_OUT_KEY][:]
    assert features.shape == (3, 9, 9, 9)
    assert features.dtype == np.float32
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features[0:3, 3:6] = 1
    expected_features[3:6, 0:3] = 1
    expected_features[3:6, 6:9] = 1
    expected_features[4, 4] = 1
    expected_features[6:9, 3:6] = 1
    assert np.equal(features[0, :, :, 4], expected_features).all()


def check_shifted_features(f: File):
    features = f[FEATURES_OUT_KEY][:]
    assert features.shape == (3, 9, 9, 9)
    assert features.dtype == np.float32
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features[0:3, 3:6] = 1
    expected_features[3:6, 0:3] = 1
    expected_features[3:6, 6:9] = 1
    expected_features[6:9, 3:6] = 1
    features_without_subject = expected_features.copy()
    expected_features[5, 5] = 1
    assert np.equal(features[0, :, :, 3], features_without_subject).all()
    assert np.equal(features[0, :, :, 4], expected_features).all()
    assert np.equal(features[0, :, :, 5], features_without_subject).all()


def check_grid_coordinates(f: File, voxel_size: int, x_min: float, x_max: float, 
                          y_min: float, y_max: float, z_min: float, z_max: float):
    """
    Validates that the coordinates dataset contains the expected grid points
    based on voxel size and extent boundaries.
    
    Parameters
    ----------
    f : File
        HDF5 file handle
    voxel_size : int
        The voxel size used for grid generation
    x_min, x_max : float
        X-axis extent boundaries
    y_min, y_max : float
        Y-axis extent boundaries
    z_min, z_max : float
        Z-axis extent boundaries
    """
    assert COORDINATES_OUT_KEY in f.keys(), f"Coordinates dataset '{COORDINATES_OUT_KEY}' not found in file"
    
    coordinates = f[COORDINATES_OUT_KEY][:]
    
    expected_x_count = int((x_max - x_min) / voxel_size) + 1
    expected_y_count = int((y_max - y_min) / voxel_size) + 1
    expected_z_count = int((z_max - z_min) / voxel_size) + 1
    
    expected_shape = (3, expected_x_count, expected_y_count, expected_z_count)
    assert coordinates.shape == expected_shape, \
        f"Coordinates shape {coordinates.shape} does not match expected {expected_shape}"
    
    assert coordinates.dtype == np.float32, \
        f"Coordinates dtype {coordinates.dtype} should be float32"
    
    x_expected = np.linspace(x_min, x_max, expected_x_count, dtype=np.float32)
    y_expected = np.linspace(y_min, y_max, expected_y_count, dtype=np.float32)
    z_expected = np.linspace(z_min, z_max, expected_z_count, dtype=np.float32)
    
    for i, x_val in enumerate(x_expected):
        assert np.allclose(coordinates[0, i, :, :], x_val), \
            f"X coordinates at index {i} should all be {x_val}"
    
    for j, y_val in enumerate(y_expected):
        assert np.allclose(coordinates[1, :, j, :], y_val), \
            f"Y coordinates at index {j} should all be {y_val}"
    
    for k, z_val in enumerate(z_expected):
        assert np.allclose(coordinates[2, :, :, k], z_val), \
            f"Z coordinates at index {k} should all be {z_val}"
    
    assert np.isclose(coordinates[0, 0, 0, 0], x_min), \
        f"Minimum X coordinate should be {x_min}"
    assert np.isclose(coordinates[0, -1, 0, 0], x_max), \
        f"Maximum X coordinate should be {x_max}"
    
    assert np.isclose(coordinates[1, 0, 0, 0], y_min), \
        f"Minimum Y coordinate should be {y_min}"
    assert np.isclose(coordinates[1, 0, -1, 0], y_max), \
        f"Maximum Y coordinate should be {y_max}"
    
    assert np.isclose(coordinates[2, 0, 0, 0], z_min), \
        f"Minimum Z coordinate should be {z_min}"
    assert np.isclose(coordinates[2, 0, 0, -1], z_max), \
        f"Maximum Z coordinate should be {z_max}"


def check_grid_attributes(f: File, voxel_size: int, x_min: float, x_max: float,
                         y_min: float, y_max: float, z_min: float, z_max: float):
    """
    Validates that the file attributes contain the correct metadata.
    
    Parameters
    ----------
    f : File
        HDF5 file handle
    voxel_size : int
        The voxel size used for grid generation
    x_min, x_max : float
        X-axis extent boundaries
    y_min, y_max : float
        Y-axis extent boundaries
    z_min, z_max : float
        Z-axis extent boundaries
    """
    assert VOXEL_SIZE_OUT_KEY in f.attrs, f"Attribute '{VOXEL_SIZE_OUT_KEY}' not found"
    assert f.attrs[VOXEL_SIZE_OUT_KEY] == voxel_size, \
        f"Voxel size attribute {f.attrs[VOXEL_SIZE_OUT_KEY]} does not match expected {voxel_size}"
    
    assert MIN_EXTENT_OUT_KEY in f.attrs, f"Attribute '{MIN_EXTENT_OUT_KEY}' not found"
    min_extent = f.attrs[MIN_EXTENT_OUT_KEY]
    expected_min = np.array([x_min, y_min, z_min], dtype=np.float32)
    assert np.allclose(min_extent, expected_min), \
        f"Min extent {min_extent} does not match expected {expected_min}"
    
    assert MAX_EXTENT_OUT_KEY in f.attrs, f"Attribute '{MAX_EXTENT_OUT_KEY}' not found"
    max_extent = f.attrs[MAX_EXTENT_OUT_KEY]
    expected_max = np.array([x_max, y_max, z_max], dtype=np.float32)
    assert np.allclose(max_extent, expected_max), \
        f"Max extent {max_extent} does not match expected {expected_max}"


def test_grid_coordinates_voxel_size_1_central(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """Test that grid coordinates are correctly generated with voxel_size=1."""
    voxel_size = 1
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    z_min, z_max = -4, 4
    
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        voxel_size=voxel_size
    )
    grid_preprocessor.process_simulations([CENTRAL_SPHERE_SIM_NAME])
    
    case_name = f"grid_voxel_size_{voxel_size}_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name
    out_simulation_file = out_dir / PROCESSED_SIMULATIONS_DIR_PATH / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    
    with File(out_simulation_file) as f:
        check_grid_coordinates(f, voxel_size, x_min, x_max, y_min, y_max, z_min, z_max)
        check_grid_attributes(f, voxel_size, x_min, x_max, y_min, y_max, z_min, z_max)


def test_grid_coordinates_voxel_size_2_shifted(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """Test that grid coordinates are correctly generated with voxel_size=2."""
    voxel_size = 2
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    z_min, z_max = -4, 4
    
    grid_preprocessor = GridPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        voxel_size=voxel_size
    )
    grid_preprocessor.process_simulations([SHIFTED_SPHERE_SIM_NAME])
    
    case_name = f"grid_voxel_size_{voxel_size}_data_type_float32"
    out_dir = processed_batch_dir_path / case_name
    out_simulation_file = out_dir / PROCESSED_SIMULATIONS_DIR_PATH / TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    
    with File(out_simulation_file) as f:
        check_grid_coordinates(f, voxel_size, x_min, x_max, y_min, y_max, z_min, z_max)
        check_grid_attributes(f, voxel_size, x_min, x_max, y_min, y_max, z_min, z_max)


def test_grid_coordinates_consistency_multiple_files(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """Test that all simulations in a batch have identical coordinates."""
    voxel_size = 1
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    z_min, z_max = -4, 4
    
    grid_preprocessor = GridPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        voxel_size=voxel_size
    )
    grid_preprocessor.process_simulations([CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME])
    
    case_name = f"grid_voxel_size_{voxel_size}_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name
    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    
    sim1_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim1_file) as f1:
        coords1 = f1[COORDINATES_OUT_KEY][:]
        check_grid_coordinates(f1, voxel_size, x_min, x_max, y_min, y_max, z_min, z_max)
    
    sim2_file = out_simulations_dir / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    with File(sim2_file) as f2:
        coords2 = f2[COORDINATES_OUT_KEY][:]
        check_grid_coordinates(f2, voxel_size, x_min, x_max, y_min, y_max, z_min, z_max)
    
    assert np.array_equal(coords1, coords2), \
        "Coordinates should be identical across all simulations in the same batch"


def test_pointcloud_float_out_dirs(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_float32"
    assert out_case_dir.exists()


def test_pointcloud_complex_out_dirs(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_complex64"
    assert out_case_dir.exists()


def test_pointcloud_antenna(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creation antenna milestones and coils mask correction.
    In details it sequentially checks the directory structure, the existence of 
    the antenna file, after opening the h5 file it checks the databases keys,
    shgapes, datatypes and the correctness of the masks.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_float32"
    out_antenna_dir = out_case_dir / PROCESSED_ANTENNA_DIR_PATH
    supposed_antenna_dir = preprop.out_antenna_dir_path
    assert supposed_antenna_dir == out_antenna_dir
    assert out_antenna_dir.exists()

    antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
    assert antenna_file.exists()

    with File(antenna_file) as f:
        assert set(f.keys()) == set([ANTENNA_MASKS_OUT_KEY])
        masks = f[ANTENNA_MASKS_OUT_KEY][:]
        assert masks.shape == (4, 729)
        assert masks.dtype == np.bool_

        assert len(np.where(masks)[0]) == 108


def test_pointcloud_general_structure_for_one_float_simulation(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creating general structure of directories and files for float data type.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_float32"
    out_sim_dir = out_case_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert preprop.out_simulations_dir_path == out_sim_dir
    assert out_sim_dir.exists()
    assert len(list(listdir(out_sim_dir))) == 1

    exact_sim_file = out_sim_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert exact_sim_file.exists()


def test_pointcloud_general_structure_for_one_complex_simulation(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creating general structure of directories and files for complex data type.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    out_case_dir = processed_batch_dir_path / "point_data_type_complex64"
    out_sim_dir = out_case_dir / PROCESSED_SIMULATIONS_DIR_PATH
    preprop.out_simulations_dir_path == out_sim_dir
    assert out_sim_dir.exists()
    assert len(list(listdir(out_sim_dir))) == 1

    exact_sim_file = out_sim_dir / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert exact_sim_file.exists()


def test_pointcloud_general_structure_for_multiple_simulations(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This testcase checks creating general structure of directories and files for multiple simulations.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME,
        CENTRAL_BOX_SIM_NAME
    ])

    out_sim_dir = preprop.out_simulations_dir_path
    assert len(list(listdir(out_sim_dir))) == 2

    first_sim_file = Path(out_sim_dir) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    assert first_sim_file.exists()

    second_sim_file = Path(out_sim_dir) / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    assert second_sim_file.exists()


def test_pointcloud_resulting_keys(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        assert set(f.keys()) == set(
            [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY, COORDINATES_OUT_KEY]
        )

        assert set(f.attrs.keys()) == set([DTYPE_OUT_KEY, TRUNCATION_COEFFICIENTS_OUT_KEY])


def test_pointcloud_resulting_dtype_values_for_float(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        assert efield.dtype == np.dtype([("re", np.float32), ("im", np.float32)])

        hfield = f[H_FIELD_OUT_KEY][:]
        assert hfield.dtype == np.dtype([("re", np.float32), ("im", np.float32)])

        features = f[FEATURES_OUT_KEY][:]
        assert features.dtype == np.float32
        
        dtype = f.attrs[DTYPE_OUT_KEY]
        assert isinstance(dtype, str)
        assert dtype == "float32"


def test_pointcloud_resulting_dtype_values_for_complex(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        assert efield.dtype == np.complex64

        hfield = f[H_FIELD_OUT_KEY][:]
        assert hfield.dtype == np.complex64

        features = f[FEATURES_OUT_KEY][:]
        assert features.dtype == np.float32

        dtype = f.attrs[DTYPE_OUT_KEY]
        assert isinstance(dtype, str)
        assert dtype == "complex64"


def test_pointcloud_datasets_shapes_and_non_changable_dtypes(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    This test case checks not the values of the resulting datasets, but shapes and dtypes.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        assert efield.shape == (3, 729, 3)

        hfield = f[H_FIELD_OUT_KEY][:]
        assert hfield.shape == (3, 729, 3)

        features = f[FEATURES_OUT_KEY][:]
        assert features.shape == (3, 729)

        coordinates = f[COORDINATES_OUT_KEY][:]
        assert coordinates.shape == (3, 729)
        assert coordinates.dtype == np.float32

        subject = f[SUBJECT_OUT_KEY][:]
        assert subject.shape == (1, 729)
        assert subject.dtype == np.bool_


def test_pointcloud_squared_coils_sphere_central_object(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Test case checks exactly the values of the resulting datasets for the 4 squared 
    coils and a central sphere.
    """
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        
        expected_field = np.concatenate(
            [
                np.zeros((3, 729, 1), dtype=np.complex64), 
                np.ones((3, 729, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(3, 729, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_squared_coils_central_box_object(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_central_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        CENTRAL_BOX_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        expected_field = np.concatenate(
            [
                np.zeros((3, 729, 1), dtype=np.complex64), 
                np.ones((3, 729, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(3, 729, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_squared_coils_shifted_sphere_object(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        SHIFTED_SPHERE_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        expected_field = np.concatenate(
            [
                np.zeros((3, 729, 1), dtype=np.complex64), 
                np.ones((3, 729, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(3, 729, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_squared_coils_shifted_box_object(raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    preprop = PointPreprocessing(
        raw_shifted_batch_dir_path,
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        coil_thick_coef=1.0
    )
    preprop.process_simulations([
        SHIFTED_BOX_SIM_NAME
    ])

    sim_file = Path(
        preprop.out_simulations_dir_path
    ) / TARGET_FILE_NAME.format(name=SHIFTED_BOX_SIM_NAME)
    with File(sim_file) as f:
        efield = f[E_FIELD_OUT_KEY][:]
        expected_field = np.concatenate(
            [
                np.zeros((3, 729, 1), dtype=np.complex64), 
                np.ones((3, 729, 1), dtype=np.complex64),
                np.full(fill_value=2, shape=(3, 729, 1), dtype=np.complex64)
            ],
            axis=-1
        )
        assert np.equal(efield, expected_field).all()

        hfield = f[H_FIELD_OUT_KEY][:]
        assert np.equal(hfield, expected_field).all()

        features = f[FEATURES_OUT_KEY][:]
        assert len(np.where(features == 1)[0]) == 327

        subject = f[SUBJECT_OUT_KEY][:]
        len(np.where(subject)[0]) == 1


def test_pointcloud_invalid_batch_path(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    with pytest.raises(FileNotFoundError):
        preprop = PointPreprocessing(
            "invalid_batch_path",
            raw_antenna_dir_path,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_pointcloud_with_empty_batch(raw_central_batch_short_term, raw_antenna_dir_path, processed_batch_dir_path):
    list(map(rmtree, raw_central_batch_short_term.iterdir()))
    with pytest.raises(FileNotFoundError):
        p = PointPreprocessing(
            raw_central_batch_short_term,
            raw_antenna_dir_path,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_pointcloud_with_no_antenna_dir(raw_central_batch_short_term, raw_antenna_dir_path_short_term, processed_batch_dir_path):
    rmtree(raw_antenna_dir_path_short_term)
    with pytest.raises(FileNotFoundError):
        p = PointPreprocessing(
            raw_central_batch_short_term,
            raw_antenna_dir_path_short_term,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_pointcloud_with_empty_antenna_dir(raw_central_batch_dir_path, raw_antenna_dir_path_short_term, processed_batch_dir_path):
    list(map(lambda x: x.unlink(), raw_antenna_dir_path_short_term.iterdir()))
    with pytest.raises(FileNotFoundError):
        p = PointPreprocessing(
            raw_central_batch_dir_path,
            raw_antenna_dir_path_short_term,
            processed_batch_dir_path,
            field_dtype=np.complex64,
            coil_thick_coef=1.0
        )


def test_multiple_batch_dirs_grid(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path, raw_shifted_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.complex64,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations()

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name
    
    assert out_dir.exists()

    assert len(list(listdir(out_dir / PROCESSED_SIMULATIONS_DIR_PATH))) == 4

    out_sim_names = listdir(out_dir / PROCESSED_SIMULATIONS_DIR_PATH)
    out_sim_names = [name.split(".")[0] for name in out_sim_names]


def test_grid_preprocessing_check_explicit_none_simulations_value(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as None explicitly
    """
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(simulations=None)

    expected_sim_list = list(natsorted(map(
        lambda x: TARGET_FILE_NAME.format(name=x.name), 
        raw_central_batch_dir_path.iterdir()
    )))
    existing_sim_list = list(natsorted(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    )))

    assert expected_sim_list == existing_sim_list


def test_grid_preprocessing_check_explicit_empty_list_simulations_value(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as an empty list explicitly
    """
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(simulations=[])

    assert len(listdir(grid_preprocessor.out_simulations_dir_path)) == 0


def test_grid_preprocessing_check_simulations_value_as_one_simulation_name_with_two_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations property as a single simulation name from one of the batches
    """
    grid_preprocessor = GridPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(simulations=[CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_grid_preprocessing_check_one_path_simulations_value_with_one_batch(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a single path from the only one existing batch
    """
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_grid_preprocessing_check_one_simulations_value_from_one_of_the_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a single simulation path from one of the batches
    """
    grid_preprocessor = GridPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_grid_preprocessing_check_multiple_paths_simulations_value_from_one_batch(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulations path from the only batch
    """
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME, raw_central_batch_dir_path / CENTRAL_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing


def test_grid_preprocessing_check_multiple_paths_simulations_value_from_different_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulations path from different batches
    """
    grid_preprocessor = GridPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME, raw_shifted_batch_dir_path / SHIFTED_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing


def test_grid_preprocessing_check_mixed_simulations_value_from_one_batch(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulation names and paths from the only batch
    """
    grid_preprocessor = GridPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        simulations=[CENTRAL_BOX_SIM_NAME, raw_central_batch_dir_path / CENTRAL_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing


def test_grid_preprocessing_check_mixed_simulations_value_from_different_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulation names and paths from different batches
    """
    grid_preprocessor = GridPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32,
        x_min=-4,
        x_max=4,
        y_min=-4,
        y_max=4,
        z_min=-4,
        z_max=4, 
        voxel_size=1
    )
    grid_preprocessor.process_simulations(
        simulations=[CENTRAL_BOX_SIM_NAME, raw_shifted_batch_dir_path / SHIFTED_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        grid_preprocessor.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing

def test_point_preprocessing_check_explicit_none_simulations_value(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as None explicitly
    """
    preprop = PointPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(simulations=None)

    expected_sim_list = list(natsorted(map(
        lambda x: TARGET_FILE_NAME.format(name=x.name), 
        raw_central_batch_dir_path.iterdir()
    )))
    existing_sim_list = list(natsorted(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    )))

    assert expected_sim_list == existing_sim_list


def test_point_preprocessing_check_explicit_empty_list_simulations_value(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as an empty list explicitly
    """
    preprop = PointPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(simulations=[])

    assert len(listdir(preprop.out_simulations_dir_path)) == 0


def test_point_preprocessing_check_simulations_value_as_one_simulation_name(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations property as a single simulation name
    """
    preprop = PointPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(simulations=[CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_point_preprocessing_check_simulations_value_as_one_simulation_name_with_two_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations property as a single simulation name from one of the batches
    """
    preprop = PointPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(simulations=[CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_point_preprocessing_check_one_path_simulations_value_with_one_batch(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a single path from the only one existing batch
    """
    preprop = PointPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_point_preprocessing_check_one_simulations_value_from_one_of_the_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a single simulation path from one of the batches
    """
    preprop = PointPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME])

    expected = [TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME)]
    existing = list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    ))

    assert expected == existing


def test_point_preprocessing_check_multiple_paths_simulations_value_from_one_batch(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulations path from the only batch
    """
    preprop = PointPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(
        simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME, raw_central_batch_dir_path / CENTRAL_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing


def test_point_preprocessing_check_multiple_paths_simulations_value_from_different_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulations path from different batches
    """
    preprop = PointPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(
        simulations=[raw_central_batch_dir_path / CENTRAL_BOX_SIM_NAME, raw_shifted_batch_dir_path / SHIFTED_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing


def test_point_preprocessing_check_mixed_simulations_value_from_one_batch(raw_central_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulation names and paths from the only batch
    """
    preprop = PointPreprocessing(
        [raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(
        simulations=[CENTRAL_BOX_SIM_NAME, raw_central_batch_dir_path / CENTRAL_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=CENTRAL_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing


def test_point_preprocessing_check_mixed_simulations_value_from_different_batches(raw_central_batch_dir_path, raw_shifted_batch_dir_path, raw_antenna_dir_path, processed_batch_dir_path):
    """
    Set simulations as a list of simulation names and paths from different batches
    """
    preprop = PointPreprocessing(
        [raw_shifted_batch_dir_path, raw_central_batch_dir_path],
        raw_antenna_dir_path,
        processed_batch_dir_path,
        field_dtype=np.float32
    )
    preprop.process_simulations(
        simulations=[CENTRAL_BOX_SIM_NAME, raw_shifted_batch_dir_path / SHIFTED_SPHERE_SIM_NAME]
    )

    expected = natsorted([
        TARGET_FILE_NAME.format(name=CENTRAL_BOX_SIM_NAME),
        TARGET_FILE_NAME.format(name=SHIFTED_SPHERE_SIM_NAME)
    ])
    existing = natsorted(list(map(
        lambda x: x.name, 
        preprop.out_simulations_dir_path.iterdir()
    )))

    assert expected == existing
