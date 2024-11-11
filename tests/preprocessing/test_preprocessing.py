from os import listdir

import pytest
import numpy as np
from h5py import File

from tests.conftest import FIRST_SIM_NAME


from magnet_pinn.preprocessing.preprocessing import (
    GridPreprocessing, PROCESSED_ANTENNA_DIR_PATH,
    PROCESSED_SIMULATIONS_DIR_PATH, TARGET_FILE_NAME,
    MASKS_DATABASE_KEY, E_FIELD_OUT_KEY, H_FIELD_OUT_KEY,
    FEATURES_OUT_KEY, SUBJECT_OUT_KEY
)


def test_grid_complex_one_simulation_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
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
        [FIRST_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_complex64"
    out_dir = processed_batch_dir_path / case_name
    assert out_dir.exists()

    check_antenna(out_dir)

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=FIRST_SIM_NAME)
    assert out_simulation_file.exists()

    assert len(list(listdir(out_simulations_dir))) == 1

    check_central_no_shifted_complex_simulation(out_simulation_file)


def check_antenna(out_dir: str):
    out_antenna_dir = out_dir / PROCESSED_ANTENNA_DIR_PATH
    assert out_antenna_dir.exists()

    out_antenna_file = out_antenna_dir / TARGET_FILE_NAME.format(name="antenna")
    assert out_antenna_file.exists()

    with File(out_antenna_file) as f:
        assert list(f.keys()) == [MASKS_DATABASE_KEY]
        masks = f[MASKS_DATABASE_KEY][:]
        assert masks.shape == (9, 9, 9, 4)
        assert masks.dtype == np.float64

        received_mask = np.sum(masks[:, :, 3, :], axis=-1)
        expected_mask = np.zeros((9, 9))
        expected_mask[3:6, 0:3] = 1
        expected_mask[3:6, 6:9] = 1
        expected_mask[0:3, 3:6] = 1
        expected_mask[6:9, 3:6] = 1
        assert np.equal(received_mask, expected_mask).all()


def check_central_no_shifted_complex_simulation(out_simulation_file: str):
    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]

        expected_field = np.zeros((3, 9, 9, 9, 1), dtype=np.complex64)

        e_field = f[E_FIELD_OUT_KEY][:]
        assert e_field.shape == (3, 9, 9, 9, 1)
        assert e_field.dtype == np.complex64
        assert np.equal(e_field, expected_field).all()

        h_field = f[H_FIELD_OUT_KEY][:]
        assert h_field.shape == (3, 9, 9, 9, 1)
        assert h_field.dtype == np.complex64
        assert np.equal(h_field, expected_field).all()

        subject_mask = f[SUBJECT_OUT_KEY][:]
        assert subject_mask.shape == (9, 9, 9, 1)
        assert subject_mask.dtype == np.float64
        expected_subject_mask = np.zeros((9, 9), dtype=np.float64)
        expected_subject_mask[3:6, 3:6] = 1
        assert np.equal(subject_mask[:, :, 3, 0], expected_subject_mask).all()

        features = f[FEATURES_OUT_KEY][:]
        assert features.shape == (3, 9, 9, 9)
        assert features.dtype == np.float32
        expected_features = np.zeros((9, 9), dtype=np.float32)
        expected_features[0:3, 3:6] = 1
        expected_features[3:6, :] = 1
        expected_features[6:9, 3:6] = 1
        assert np.equal(features[0, :, :, 3], expected_features).all()


def test_grid_float_one_simulation_valid_preprocessing(raw_batch_dir_path, processed_batch_dir_path):
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
        [FIRST_SIM_NAME]
    )

    case_name = f"grid_voxel_size_1_data_type_float32"
    out_dir = processed_batch_dir_path / case_name
    assert out_dir.exists()

    check_antenna(out_dir)

    out_simulations_dir = out_dir / PROCESSED_SIMULATIONS_DIR_PATH
    assert out_simulations_dir.exists()

    out_simulation_file = out_simulations_dir / TARGET_FILE_NAME.format(name=FIRST_SIM_NAME)
    assert out_simulation_file.exists()

    assert len(list(listdir(out_simulations_dir))) == 1

    check_central_no_shifted_float_simulation(out_simulation_file)


def check_central_no_shifted_float_simulation(out_simulation_file: str):
    with File(out_simulation_file) as f:
        list(f.keys()) == [E_FIELD_OUT_KEY, H_FIELD_OUT_KEY, FEATURES_OUT_KEY, SUBJECT_OUT_KEY]

        expected_field = np.zeros((3, 9, 9, 9, 1), dtype=np.float32)

        e_field = f[E_FIELD_OUT_KEY][:]
        assert e_field.shape == (3, 9, 9, 9, 1)
        assert e_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
        re = e_field["re"]
        assert np.equal(re, expected_field).all()
        im = e_field["im"]
        assert np.equal(im, expected_field).all()

        h_field = f[H_FIELD_OUT_KEY][:]
        assert h_field.shape == (3, 9, 9, 9, 1)
        assert h_field.dtype == np.dtype([("re", np.float32), ("im", np.float32)])
        re = h_field["re"]
        assert np.equal(re, expected_field).all()
        im = h_field["im"]
        assert np.equal(im, expected_field).all()

        subject_mask = f[SUBJECT_OUT_KEY][:]
        assert subject_mask.shape == (9, 9, 9, 1)
        assert subject_mask.dtype == np.float64
        expected_subject_mask = np.zeros((9, 9), dtype=np.float64)
        expected_subject_mask[3:6, 3:6] = 1
        assert np.equal(subject_mask[:, :, 3, 0], expected_subject_mask).all()

        features = f[FEATURES_OUT_KEY][:]
        assert features.shape == (3, 9, 9, 9)
        assert features.dtype == np.float32
        expected_features = np.zeros((9, 9), dtype=np.float32)
        expected_features[0:3, 3:6] = 1
        expected_features[3:6, :] = 1
        expected_features[6:9, 3:6] = 1
        assert np.equal(features[0, :, :, 3], expected_features).all()

