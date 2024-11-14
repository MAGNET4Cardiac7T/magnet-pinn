from os import listdir

import pytest
import numpy as np
from h5py import File

from tests.conftest import (
    CENTRAL_SPHERE_SIM_NAME, CENTRAL_BOX_SIM_NAME,
    SHIFTED_BOX_SIM_NAME, SHIFTED_SPHERE_SIM_NAME
)


from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PROCESSED_ANTENNA_DIR_PATH,
    PROCESSED_SIMULATIONS_DIR_PATH, TARGET_FILE_NAME,
    ANTENNA_MASKS_OUT_KEY, E_FIELD_OUT_KEY, H_FIELD_OUT_KEY,
    FEATURES_OUT_KEY, SUBJECT_OUT_KEY
)

def test_antenna_processing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
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
    assert out_dir.exists()

    check_antenna(out_dir)


def test_grid_central_complex_one_simulation_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
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


def test_grid_central_complex_multiple_simulations_valid_preprcessing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
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
        assert masks.shape == (9, 9, 9, 4)
        assert masks.dtype == np.bool_

        received_mask = np.sum(masks[:, :, 3, :], axis=-1)
        expected_mask = np.zeros((9, 9))
        expected_mask[3:6, 0:3] = 1
        expected_mask[3:6, 6:9] = 1
        expected_mask[0:3, 3:6] = 1
        expected_mask[6:9, 3:6] = 1
        assert np.equal(received_mask, expected_mask).all()


def test_grid_central_float_one_simulation_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
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


def test_grid_central_float_multiple_simulations_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
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


def test_grid_shifted_one_simulation_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
    grid_preprocessor = GridPreprocessing(
        raw_batch_dir_path,
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
    expected_field = np.zeros((3, 9, 9, 9, 1), dtype=np.complex64)

    e_field = f[E_FIELD_OUT_KEY][:]
    assert e_field.shape == (3, 9, 9, 9, 1)
    assert e_field.dtype == np.complex64
    assert np.equal(e_field, expected_field).all()

    h_field = f[H_FIELD_OUT_KEY][:]
    assert h_field.shape == (3, 9, 9, 9, 1)
    assert h_field.dtype == np.complex64
    assert np.equal(h_field, expected_field).all()


def check_float_fields(f: File):
    expected_field = np.zeros((3, 9, 9, 9, 1), dtype=np.float32)

    e_field = f[E_FIELD_OUT_KEY][:]
    assert e_field.shape == (3, 9, 9, 9, 1)
    assert e_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
    re_e_field = e_field["re"]
    assert np.equal(re_e_field, expected_field).all()
    im_e_field = e_field["im"]
    assert np.equal(im_e_field, expected_field).all()

    h_field = f[H_FIELD_OUT_KEY][:]
    assert h_field.shape == (3, 9, 9, 9, 1)
    assert h_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
    re_h_field = h_field["re"]
    assert np.equal(re_h_field, expected_field).all()
    im_h_field = h_field["im"]
    assert np.equal(im_h_field, expected_field).all()


def check_central_subject_mask(f: File):
    subject_mask = f[SUBJECT_OUT_KEY][:]
    assert subject_mask.shape == (9, 9, 9, 1)
    assert subject_mask.dtype == np.bool_
    expected_subject_mask = np.zeros((9, 9), dtype=np.bool_)
    expected_subject_mask[3:6, 3:6] = 1
    assert np.equal(subject_mask[:, :, 3, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 4, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 5, 0], expected_subject_mask).all()

def check_shifted_subject_mask(f: File):
    subject_mask = f[SUBJECT_OUT_KEY][:]
    assert subject_mask.shape == (9, 9, 9, 1)
    assert subject_mask.dtype == np.bool_
    expected_subject_mask = np.zeros((9, 9), dtype=np.bool_)
    expected_subject_mask[4:7, 4:7] = 1
    assert np.equal(subject_mask[:, :, 3, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 4, 0], expected_subject_mask).all()
    assert np.equal(subject_mask[:, :, 5, 0], expected_subject_mask).all()


def check_central_features(f: File):
    features = f[FEATURES_OUT_KEY][:]
    assert features.shape == (3, 9, 9, 9)
    assert features.dtype == np.float32
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features[0:3, 3:6] = 1
    expected_features[3:6, :] = 1
    expected_features[6:9, 3:6] = 1
    assert np.equal(features[0, :, :, 3], expected_features).all()


def check_shifted_features(f: File):
    features = f[FEATURES_OUT_KEY][:]
    assert features.shape == (3, 9, 9, 9)
    assert features.dtype == np.float32
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features = np.zeros((9, 9), dtype=np.float32)
    expected_features[0:3, 3:6] = 1
    expected_features[3:6, 0:3] = 1
    expected_features[3:6, 6:9] = 1
    expected_features[4:7, 4:7] = 1
    expected_features[6:9, 3:6] = 1
    expected_features[6, 4:6] = 2
    expected_features[4:6, 6] = 2
    assert np.equal(features[0, :, :, 3], expected_features).all()
    assert np.equal(features[0, :, :, 4], expected_features).all()
    assert np.equal(features[0, :, :, 5], expected_features).all()